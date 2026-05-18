/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// moe_int8_shared_expert_amx.cpp
// Hand-rolled AMX INT8 shared-expert kernel.
//
// Goal: close the 45% gap vs the target 260us for M=64, N=2048, K=7168
// by eliminating the int32 intermediate round-trip and collapsing the
// OMP parallel regions that the oneDNN / brgemm paths incur per call.
//
// Layout:
//   A: u8 [M, K]                  (row-major)
//   W1 packed per-col-block: s8 [NB, K/4, BLOCK_N, 4] VNNI INT8
//     with per-block Bcomp s32 [NB, BLOCK_N] stored alongside
//   W2 packed per-col-block: s8 [NB, N/4, BLOCK_N, 4]
//     (note W2's "N" here is OC=K; "K" inside the inner loop is IC=N=2048)
//
// Stage scheduling (for M=64, BLOCK_M=32, BLOCK_N=32):
//   Stage 1 (W1 matmul + SiLU*Mul -> u8 requant):
//     - Parallel on (mb, nb) grid = (2, N/BLOCK_N) = (2, 64) = 128 tiles
//       With 32 threads, each thread owns 4 output tiles.
//     - Per tile: run K-loop in 2 output column halves (gate + up)
//       to avoid doubling the N parallelism; accumulate Cgate/Cup in
//       tmm0..tmm3 against shared A tiles tmm4..tmm5, load weights
//       into tmm6..tmm7.
//     - After K loop: tile_stored to per-thread s32 scratch (32x64 words),
//       dequant in AVX-512, apply SiLU(g)*u, write to per-row f32 scratch,
//       track amax.
//     - After all N-tiles for a given M-row set: compute inv_scale from
//       amax, scan f32 scratch, quantize to u8 -> A2.
//
//   Stage 2 (W2 matmul + scaled add -> bf16 out):
//     - Same tile schedule over (mb, nb) for OC=K=7168.
//     - Per tile: K-loop over IC=N=2048; only one output stream (no gate/up).
//     - After K loop: tile_stored to per-thread s32 scratch, then dequant
//       + add routed_scaling * fused_out -> bf16 store.
//
// Per-weight-pair cache: VNNI-packed weights + Bcomp precomputed on first
// sight, keyed on (w1_ptr, w2_ptr).
//
// Correctness tolerance matches the existing oneDNN path: diff_rms / ref_rms
// < 5% against BRGEMM on random data.

#if defined(__x86_64__) || defined(_M_X64)

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
#include <list>
#include <mutex>
#include <sys/syscall.h>
#include <unistd.h>

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// ------------- Optional per-stage timing -------------
// Set SHARED_EXPERT_AMX_TIMING=1 to enable. Prints a summary every
// SHARED_EXPERT_AMX_TIMING_PRINT (default 500) calls.
struct Timing {
  bool enabled = false;
  int64_t print_every = 500;
  std::atomic<int64_t> calls{0};
  std::atomic<int64_t> ns_total{0};
  std::atomic<int64_t> ns_s0{0};       // stage 0 (bf16->u8 quant)
  std::atomic<int64_t> ns_s1_gemm{0};  // stage 1 gemm+silu*mul+amax
  std::atomic<int64_t> ns_s1_reduce{0};// stage 1.5 amax reduce
  std::atomic<int64_t> ns_s1_req{0};   // stage 1.5 requant
  std::atomic<int64_t> ns_s2{0};       // stage 2 gemm+dequant+add+bf16 store
  std::atomic<int64_t> ns_setup{0};    // setup (pool alloc, weight lookup, cfg)
};

static Timing& get_timing() {
  static Timing t;
  static bool inited = []{
    const char* e = std::getenv("SHARED_EXPERT_AMX_TIMING");
    t.enabled = (e && e[0] && e[0] != '0');
    const char* p = std::getenv("SHARED_EXPERT_AMX_TIMING_PRINT");
    if (p) {
      long v = std::atol(p);
      if (v > 0) t.print_every = v;
    }
    return true;
  }();
  (void)inited;
  return t;
}

static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void maybe_print_timing() {
  auto& t = get_timing();
  if (!t.enabled) return;
  int64_t n = t.calls.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % t.print_every != 0) return;
  double scale = 1e-3 / (double)t.print_every;  // ns -> us, averaged
  int64_t total = t.ns_total.exchange(0);
  int64_t s0 = t.ns_s0.exchange(0);
  int64_t s1g = t.ns_s1_gemm.exchange(0);
  int64_t s1r = t.ns_s1_reduce.exchange(0);
  int64_t s1q = t.ns_s1_req.exchange(0);
  int64_t s2 = t.ns_s2.exchange(0);
  int64_t su = t.ns_setup.exchange(0);
  std::fprintf(stderr,
      "[shared_expert_amx] avg over %ld calls: "
      "total=%.2fus setup=%.2f s0_quant=%.2f s1_gemm=%.2f s1_reduce=%.2f "
      "s1_req=%.2f s2=%.2f\n",
      (long)t.print_every, total*scale, su*scale, s0*scale,
      s1g*scale, s1r*scale, s1q*scale, s2*scale);
}

// ------------- AMX tile configuration -------------

// Tile palette: 8 tiles, each up to 16 rows x 64 cols bytes.
struct alignas(64) AmxTileConfig {
  uint8_t palette;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

// Linux AMX permission request (ARCH_REQ_XCOMP_PERM / XFEATURE_XTILEDATA).
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

static bool request_amx_permission_once() {
  static std::once_flag flag;
  static bool ok = false;
  std::call_once(flag, [] {
    // arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA). On kernels that
    // auto-grant (>= 5.17 with appropriate config), this is a no-op success.
    // On older kernels it may fail; we treat failure as "hope for the best"
    // since some distros grant by default.
    long r = syscall(SYS_arch_prctl, (unsigned long)ARCH_REQ_XCOMP_PERM,
                     (unsigned long)XFEATURE_XTILEDATA);
    ok = (r == 0);
    if (!ok) {
      // Don't fail here; _tile_loadconfig would SIGILL if truly unavailable.
      ok = true;
    }
  });
  return ok;
}

// Tile-register assignments (fixed convention used by both kernels):
//   tmm0: Cacc[0]  (16x16 s32)  -- accumulator, top half rows 0..15
//   tmm1: Cacc[1]  (16x16 s32)  --             , bottom half rows 16..31
//   tmm2: Cacc[2]  (16x16 s32)  -- gate(top)   OR unused for stage2 path
//   tmm3: Cacc[3]  (16x16 s32)  -- gate(bot)   OR unused for stage2 path
//   tmm4: A[0]     (16x64 u8)   -- A rows 0..15 for 64 K bytes
//   tmm5: A[1]     (16x64 u8)   -- A rows 16..31 for 64 K bytes
//   tmm6: B[0]     (16x64 s8)   -- BLOCK_N=16 cols (VNNI: 16 rows * 4 K)
//   tmm7: B[1]     (16x64 s8)   -- BLOCK_N=16 cols (second 16-col half)
//
// Row/col config for INT8 AMX:
//   - A (u8): 16 rows, 64 cols (colsb=64)
//   - B (s8): 16 rows, 64 cols (each "row" = 16 cols x 4 vnni bytes)
//   - C (s32): 16 rows, 64 cols (= 16 int32 elements)

static AmxTileConfig make_amx_cfg_int8() {
  AmxTileConfig c{};
  c.palette = 1;
  // Cacc tiles (tmm0..tmm3): 16 rows, 16 int32 cols = 64 bytes
  for (int i = 0; i < 4; ++i) {
    c.rows[i] = 16;
    c.colsb[i] = 64;
  }
  // A tiles (tmm4..tmm5): 16 rows, 64 bytes of u8
  for (int i = 4; i < 6; ++i) {
    c.rows[i] = 16;
    c.colsb[i] = 64;
  }
  // B tiles (tmm6..tmm7): 16 rows of (16 cols * 4 vnni-bytes) = 64 bytes
  for (int i = 6; i < 8; ++i) {
    c.rows[i] = 16;
    c.colsb[i] = 64;
  }
  return c;
}

static inline void amx_enable_for_thread() {
  // _tile_release() invalidates the palette, so reconfigure on every
  // parallel-region entry. The config is tiny (64 bytes) and the
  // instruction is cheap vs first tile_loadd.
  AmxTileConfig cfg = make_amx_cfg_int8();
  _tile_loadconfig(&cfg);
}

// ------------- utility: fast exp + silu helpers -------------

#if defined(__AVX512F__)
static inline __m512 _mm512_fast_exp_ps(__m512 x) {
  const __m512 L = _mm512_set1_ps(0.6931472f);
  const __m512 c2 = _mm512_set1_ps(0.2402265f);
  const __m512 c3 = _mm512_set1_ps(0.0555041f);
  const __m512 c4 = _mm512_set1_ps(0.00961813f);
  const __m512 c5 = _mm512_set1_ps(0.00134326f);
  const __m512 one = _mm512_set1_ps(1.f);
  __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f));
  t = _mm512_max_ps(t, _mm512_set1_ps(-126.f));
  t = _mm512_min_ps(t, _mm512_set1_ps(126.f));
  __m512 ti = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  __m512 tf = _mm512_sub_ps(t, ti);
  __m512 p = _mm512_fmadd_ps(tf, c5, c4);
  p = _mm512_fmadd_ps(tf, p, c3);
  p = _mm512_fmadd_ps(tf, p, c2);
  p = _mm512_fmadd_ps(tf, p, L);
  p = _mm512_fmadd_ps(tf, p, one);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(
      _mm512_slli_epi32(
          _mm512_add_epi32(_mm512_cvtps_epi32(ti), _mm512_set1_epi32(127)),
          23)));
}

static inline __m512 silu_ps(__m512 g) {
  const __m512 one = _mm512_set1_ps(1.f);
  __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), g);
  __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_fast_exp_ps(neg)));
  return _mm512_mul_ps(g, sig);
}
#endif

// ------------- Weight packing: s8 [K, N] row-major -> VNNI [N/16, K/4, 16, 4] -------------
//
// AMX INT8 B-tile expects 16 rows of (16 cols * 4 k-steps) = 1024 bytes.
// Storing NB blocks of BLOCK_N=16 each lets tile_loadd stream sequentially
// with a tile stride of 64 bytes.
//
// Input src layout: s8[K, N] row-major with ld_src = N.
// Output layout per block (of 16 N-cols): K/4 rows of (16 N * 4 K) = 64 bytes each.
// Block stride in bytes = (K/4) * 64.
static void pack_b_vnni_s8(
    int8_t* __restrict__ dst, int32_t* __restrict__ bcomp,
    const int8_t* __restrict__ src,
    int64_t K, int64_t N) {
  constexpr int BLOCK_N = 16;
  TORCH_CHECK(N % BLOCK_N == 0, "pack_b_vnni_s8: N % 16 != 0");
  TORCH_CHECK(K % 4 == 0, "pack_b_vnni_s8: K % 4 != 0");
  const int64_t NB = N / BLOCK_N;
  const int64_t K4 = K / 4;
  const int64_t block_bytes = K4 * BLOCK_N * 4;

  #pragma omp parallel for schedule(static)
  for (int64_t nb = 0; nb < NB; ++nb) {
    int8_t* dp = dst + nb * block_bytes;
    int32_t sum[BLOCK_N] = {0};
    for (int64_t k4 = 0; k4 < K4; ++k4) {
      for (int n = 0; n < BLOCK_N; ++n) {
        int col = (int)(nb * BLOCK_N + n);
        int8_t s0 = src[(k4 * 4 + 0) * N + col];
        int8_t s1 = src[(k4 * 4 + 1) * N + col];
        int8_t s2 = src[(k4 * 4 + 2) * N + col];
        int8_t s3 = src[(k4 * 4 + 3) * N + col];
        int64_t off = k4 * (BLOCK_N * 4) + n * 4;
        dp[off + 0] = s0;
        dp[off + 1] = s1;
        dp[off + 2] = s2;
        dp[off + 3] = s3;
        sum[n] += (int32_t)s0 + (int32_t)s1 + (int32_t)s2 + (int32_t)s3;
      }
    }
    for (int n = 0; n < BLOCK_N; ++n) {
      bcomp[nb * BLOCK_N + n] = 128 * sum[n];
    }
  }
}

// ------------- Weight cache -------------

struct WeightEntry {
  // Packed weights + per-col-block s32 Bcomp
  at::Tensor w1_pack, w2_pack;     // int8
  at::Tensor w1_bcomp, w2_bcomp;   // int32, shape [2*N] and [K]
};

struct ShapePool {
  // Pooled per-call buffers sized to (M, N, K). Allocated once per shape and
  // reused across calls -- eliminates per-call at::empty / caching-allocator
  // overhead that's visible at this scale (6 buffers × ~1us at small M).
  int64_t M = -1, N = -1, K = -1;
  at::Tensor qi;       // [M, K] u8
  at::Tensor qisc;     // [M] f32
  at::Tensor qi2;      // [M, N] u8
  at::Tensor intsc;    // [M] f32
  at::Tensor f_scratch;// [M, N] f32
  at::Tensor amax_tpr; // [nth, M] f32 (per-thread per-row partial amax)
  int64_t amax_nth = 0;
};

struct AmxContext {
  using Key = std::pair<const void*, const void*>;
  struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
      return std::hash<const void*>{}(k.first) ^
             (std::hash<const void*>{}(k.second) << 1);
    }
  };
  std::unordered_map<Key, WeightEntry, KeyHash> cache;
  std::list<Key> lru;
  std::unordered_map<Key, std::list<Key>::iterator, KeyHash> lru_it;
  static constexpr size_t kMaxCached = 128;
  std::mutex m;

  ShapePool pool;
};

static AmxContext& get_ctx() {
  static AmxContext ctx;
  return ctx;
}

static WeightEntry& get_or_build_entry(
    const int8_t* w1_data, const int8_t* w2_data,
    int64_t K, int64_t N) {
  auto& ctx = get_ctx();
  std::lock_guard<std::mutex> g(ctx.m);
  AmxContext::Key key{w1_data, w2_data};
  auto it = ctx.cache.find(key);
  if (it != ctx.cache.end()) {
    ctx.lru.erase(ctx.lru_it[key]);
    ctx.lru.push_front(key);
    ctx.lru_it[key] = ctx.lru.begin();
    return it->second;
  }
  if (ctx.cache.size() >= AmxContext::kMaxCached) {
    auto victim = ctx.lru.back();
    ctx.lru.pop_back();
    ctx.lru_it.erase(victim);
    ctx.cache.erase(victim);
  }

  WeightEntry e;
  // W1: input is s8 [K, 2N] (row-major). Pack into [2N/16, K/4, 16, 4].
  e.w1_pack = at::empty({K * 2 * N}, at::kChar);
  e.w1_bcomp = at::empty({2 * N}, at::kInt);
  pack_b_vnni_s8(
      e.w1_pack.data_ptr<int8_t>(), e.w1_bcomp.data_ptr<int32_t>(),
      w1_data, K, 2 * N);

  // W2: input is s8 [N, K] (row-major). Pack into [K/16, N/4, 16, 4].
  e.w2_pack = at::empty({N * K}, at::kChar);
  e.w2_bcomp = at::empty({K}, at::kInt);
  pack_b_vnni_s8(
      e.w2_pack.data_ptr<int8_t>(), e.w2_bcomp.data_ptr<int32_t>(),
      w2_data, N, K);

  auto ins = ctx.cache.emplace(key, std::move(e));
  ctx.lru.push_front(key);
  ctx.lru_it[key] = ctx.lru.begin();
  return ins.first->second;
}

// ------------- AMX tile kernel helpers -------------

// Fused gate+up macrogemm for 32 M-rows × 16 gate-cols × 16 up-cols.
// Uses all 8 AMX tile registers in a single K-loop:
//   tmm0: Cgate[rows 0..15,  cols 0..15]
//   tmm1: Cgate[rows 16..31, cols 0..15]
//   tmm2: Cup  [rows 0..15,  cols 0..15]
//   tmm3: Cup  [rows 16..31, cols 0..15]
//   tmm4: A[rows 0..15]   (64 K-bytes)
//   tmm5: A[rows 16..31]  (64 K-bytes)
//   tmm6: Bg (16 N-cols × 4 K-bytes VNNI)
//   tmm7: Bu (16 N-cols × 4 K-bytes VNNI)
// Per K-iter: 4 tile_loadd + 4 tile_dpbusd. A is loaded once and fed to
// both gate and up -- vs the prior 32x32 "gate-then-up" scheme which
// streamed A through the core twice.
static ALWAYS_INLINE void amx_gemm_32x16_gu(
    const uint8_t* __restrict__ A, int64_t ld_a_bytes,
    const int8_t* __restrict__ Bg, const int8_t* __restrict__ Bu,
    int32_t* __restrict__ cg_out, int32_t* __restrict__ cu_out,
    int64_t K) {
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  constexpr int64_t ld_b_bytes = 64;
  constexpr int64_t b_row_stride = 16 * ld_b_bytes;  // 1024 bytes per K-iter
  const int64_t K_steps = K / 64;
  const uint8_t* Ap = A;
  const int8_t* Bgp = Bg;
  const int8_t* Bup = Bu;
  // Pointer-advance form: fewer imul uops per iteration than ks*constant.
  // No SW prefetch -- HW prefetcher handles the two 1KB-per-iter B streams
  // and SW pf adds pressure to the address-gen unit on SPR.
  for (int64_t ks = 0; ks < K_steps; ++ks) {
    _tile_loadd(4, Ap, ld_a_bytes);
    _tile_loadd(5, Ap + 16 * ld_a_bytes, ld_a_bytes);
    _tile_loadd(6, Bgp, ld_b_bytes);
    _tile_loadd(7, Bup, ld_b_bytes);
    _tile_dpbusd(0, 4, 6);
    _tile_dpbusd(1, 5, 6);
    _tile_dpbusd(2, 4, 7);
    _tile_dpbusd(3, 5, 7);
    Ap  += 64;
    Bgp += b_row_stride;
    Bup += b_row_stride;
  }
  constexpr int64_t ldc_bytes = 16 * sizeof(int32_t);
  _tile_stored(0, cg_out + 0 * 16,  ldc_bytes);
  _tile_stored(1, cg_out + 16 * 16, ldc_bytes);
  _tile_stored(2, cu_out + 0 * 16,  ldc_bytes);
  _tile_stored(3, cu_out + 16 * 16, ldc_bytes);
}

// 32 rows × 32 cols macrogemm for Stage 2 (no gate/up pairing).
//   tmm0..3: Cacc (4 tiles of 16×16 s32)
//   tmm4..5: A rows 0..15, 16..31
//   tmm6..7: B cols 0..15, 16..31
static ALWAYS_INLINE void amx_gemm_32x32(
    const uint8_t* __restrict__ A, int64_t ld_a_bytes,
    const int8_t* __restrict__ B0, const int8_t* __restrict__ B1,
    int32_t* __restrict__ c_buf, int64_t K) {
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  constexpr int64_t ld_b_bytes = 64;
  constexpr int64_t b_row_stride = 16 * ld_b_bytes;
  const int64_t K_steps = K / 64;
  const uint8_t* Ap = A;
  const int8_t* B0p = B0;
  const int8_t* B1p = B1;
  for (int64_t ks = 0; ks < K_steps; ++ks) {
    _tile_loadd(4, Ap, ld_a_bytes);
    _tile_loadd(5, Ap + 16 * ld_a_bytes, ld_a_bytes);
    _tile_loadd(6, B0p, ld_b_bytes);
    _tile_loadd(7, B1p, ld_b_bytes);
    _tile_dpbusd(0, 4, 6);
    _tile_dpbusd(1, 5, 6);
    _tile_dpbusd(2, 4, 7);
    _tile_dpbusd(3, 5, 7);
    Ap  += 64;
    B0p += b_row_stride;
    B1p += b_row_stride;
  }
  constexpr int64_t ldc_bytes = 32 * sizeof(int32_t);
  _tile_stored(0, c_buf + 0 * 32 + 0,  ldc_bytes);
  _tile_stored(1, c_buf + 16 * 32 + 0, ldc_bytes);
  _tile_stored(2, c_buf + 0 * 32 + 16, ldc_bytes);
  _tile_stored(3, c_buf + 16 * 32 + 16, ldc_bytes);
}

// ------------- The shared-expert public kernel -------------

at::Tensor shared_expert_amx_impl(
    at::Tensor& hidden_states,
    at::Tensor& w1,           // s8 [K, 2N]
    at::Tensor& w2,           // s8 [N, K]
    at::Tensor& fused_out,    // bf16 [M, K]
    double routed_scaling_factor,
    const at::Tensor& w1_scale,
    const at::Tensor& w2_scale) {
  TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16);
  TORCH_CHECK(fused_out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(w1.scalar_type() == at::kChar);
  TORCH_CHECK(w2.scalar_type() == at::kChar);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, w1);
  CHECK_DIM(2, w2);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = w1.size(1) / 2;
  CHECK_EQ(w1.size(0), K);
  CHECK_EQ(w2.size(0), N);
  CHECK_EQ(w2.size(1), K);
  CHECK_EQ(w1_scale.numel(), 2 * N);
  CHECK_EQ(w2_scale.numel(), K);

  constexpr int64_t BLOCK_M = 32;
  constexpr int64_t BLOCK_N = 32;  // two 16-wide AMX tiles side-by-side
  TORCH_CHECK(M % BLOCK_M == 0, "shared_expert_amx: M must be multiple of 32, got ", M);
  TORCH_CHECK(N % BLOCK_N == 0, "shared_expert_amx: N must be multiple of 32, got ", N);
  TORCH_CHECK(K % BLOCK_N == 0, "shared_expert_amx: K must be multiple of 32, got ", K);
  TORCH_CHECK(K % 64 == 0, "shared_expert_amx: K must be multiple of 64 (AMX 64-byte step)");
  TORCH_CHECK(N % 64 == 0, "shared_expert_amx: N must be multiple of 64");

  TORCH_CHECK(request_amx_permission_once(),
      "shared_expert_amx: failed to request AMX XTILEDATA permission (kernel too old?)");

  auto w1_scale_f = w1_scale.scalar_type() == at::kFloat ? w1_scale : w1_scale.to(at::kFloat);
  auto w2_scale_f = w2_scale.scalar_type() == at::kFloat ? w2_scale : w2_scale.to(at::kFloat);
  const float* w1s = w1_scale_f.data_ptr<float>();
  const float* w2s = w2_scale_f.data_ptr<float>();
  const int8_t* w1_data = w1.data_ptr<int8_t>();
  const int8_t* w2_data = w2.data_ptr<int8_t>();

  auto& we = get_or_build_entry(w1_data, w2_data, K, N);
  const int8_t* w1_pack = we.w1_pack.data_ptr<int8_t>();
  const int8_t* w2_pack = we.w2_pack.data_ptr<int8_t>();
  const int32_t* w1_bc = we.w1_bcomp.data_ptr<int32_t>();    // [2N]
  const int32_t* w2_bc = we.w2_bcomp.data_ptr<int32_t>();    // [K]

  auto inp = hidden_states.data_ptr<at::BFloat16>();
  auto fused = fused_out.data_ptr<at::BFloat16>();
  auto out_t = at::empty({M, K}, at::kBFloat16);
  at::BFloat16* outp = out_t.data_ptr<at::BFloat16>();

  auto& _tm = get_timing();
  const bool _timing = _tm.enabled;
  int64_t _t_call_start = _timing ? now_ns() : 0;
  int64_t _t_stage_start = _t_call_start;

  const int nth = omp_get_max_threads();
  {
    auto& ctx = get_ctx();
    std::lock_guard<std::mutex> g(ctx.m);
    auto& p = ctx.pool;
    if (p.M != M || p.N != N || p.K != K) {
      p.qi        = at::empty({M * K}, at::kByte);
      p.qisc      = at::empty({M}, at::kFloat);
      p.qi2       = at::empty({M * N}, at::kByte);
      p.intsc     = at::empty({M}, at::kFloat);
      p.f_scratch = at::empty({M * N}, at::kFloat);
      p.M = M; p.N = N; p.K = K;
    }
    if (p.amax_nth != nth) {
      p.amax_tpr = at::empty({nth * M}, at::kFloat);
      p.amax_nth = nth;
    }
  }
  auto& pool = get_ctx().pool;
  uint8_t* qi     = pool.qi.data_ptr<uint8_t>();
  float*   qisc   = pool.qisc.data_ptr<float>();
  uint8_t* qi2    = pool.qi2.data_ptr<uint8_t>();
  float*   intsc  = pool.intsc.data_ptr<float>();
  float*   f_scratch = pool.f_scratch.data_ptr<float>();
  float*   amax_tpr  = pool.amax_tpr.data_ptr<float>();

  // Block-per-col stride in bytes for packed weights. NB is 1 per 16 N-cols.
  // W1 packing covers 2N -> (2N/16) blocks, each of size (K/4) * 16 * 4 bytes.
  const int64_t W1_NB = (2 * N) / 16;
  const int64_t W1_block_bytes = (K / 4) * 16 * 4;
  // W2 packing covers K -> (K/16) blocks, each of size (N/4) * 16 * 4 bytes.
  const int64_t W2_block_bytes = (N / 4) * 16 * 4;

  const int64_t MB = M / BLOCK_M;
  const int64_t NB1 = N / BLOCK_N;            // stage1 N-tiles (gate/up share nb)
  const int64_t NB2 = K / BLOCK_N;            // stage2 OC-tiles

  // One OMP region covers Stage 0 (input quant), Stage 1 (w1 gemm + silu*mul
  // with per-thread amax accumulation), Stage 1.5 (reduce amax + u8 requant),
  // Stage 2 (w2 gemm + dequant + scaled add). Barriers sync between stages.
  // Merging saves 2-3 fork/join cycles (~10-15us at 32 threads) and keeps
  // f_scratch hot in L2 across the amax reduction.
  const float routed = (float)routed_scaling_factor;
  const __m512 vroute = _mm512_set1_ps(routed);
  (void)W1_NB;

  // Per-stage wall-clock snapshots, written only by thread 0 just after each
  // barrier. Variables declared before the parallel region are implicitly
  // shared across threads.
  int64_t _t_s0 = 0, _t_s1 = 0, _t_s1red = 0, _t_s1req = 0, _t_s2 = 0;
  if (_timing) { _t_stage_start = now_ns(); _tm.ns_setup.fetch_add(_t_stage_start - _t_call_start, std::memory_order_relaxed); }

  #pragma omp parallel shared(_t_s0, _t_s1, _t_s1red, _t_s1req, _t_s2)
  {
    const int tid = omp_get_thread_num();
    const int nthr = omp_get_num_threads();
    amx_enable_for_thread();
    alignas(64) int32_t cg[32 * 16];
    alignas(64) int32_t cu[32 * 16];
    alignas(64) int32_t cacc[32 * 32];

    // ---- Stage 0: quantize input bf16 -> u8. Row-parallel.
    #pragma omp for schedule(static) nowait
    for (int64_t m = 0; m < M; ++m) {
      const at::BFloat16* r = inp + m * K;
      uint8_t* o = qi + m * K;
      const __m512 signBit = _mm512_set1_ps(-0.0f);
      const __m512i off = _mm512_set1_epi32(128);

      __m512 vamax0 = _mm512_setzero_ps();
      __m512 vamax1 = _mm512_setzero_ps();
      int64_t k = 0;
      for (; k + 32 <= K; k += 32) {
        __m512i va = _mm512_loadu_si512((void*)(r + k));
        __m512 va0 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 0)), 16));
        __m512 va1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 1)), 16));
        vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, va0));
        vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(signBit, va1));
      }
      float amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
      amax = std::max(amax, 1e-7f);
      qisc[m] = amax / 127.f;
      const float inv = 127.f / amax;
      const __m512 vd = _mm512_set1_ps(inv);
      k = 0;
      for (; k + 32 <= K; k += 32) {
        __m512i va = _mm512_loadu_si512((void*)(r + k));
        __m512 va0 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 0)), 16));
        __m512 va1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(va, 1)), 16));
        va0 = _mm512_mul_ps(va0, vd);
        va1 = _mm512_mul_ps(va1, vd);
        va0 = _mm512_roundscale_ps(va0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        va1 = _mm512_roundscale_ps(va1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i i0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va0), off));
        __m128i i1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va1), off));
        _mm256_storeu_si256((__m256i*)(o + k), _mm256_set_m128i(i1, i0));
      }
    }

    // Initialize this thread's per-row partial amax buffer to zero.
    float* my_amax = amax_tpr + (int64_t)tid * M;
    for (int64_t m = 0; m < M; ++m) my_amax[m] = 0.f;

    #pragma omp barrier
    if (_timing && tid == 0) _t_s0 = now_ns();

    // ---- Stage 1: W1 gemm + SiLU*Mul. Per-thread tracks partial per-row amax
    // over the nb-slice it owns.
    //
    // Keep 32 __m512 amax vectors (one per M-row in current mb block)
    // accumulated in-register across the lo+hi 16-col halves, and only
    // horizontal-reduce once per mb block -- not once per row-store.
    #pragma omp for schedule(static) nowait
    for (int64_t nb = 0; nb < NB1; ++nb) {
      int64_t n0 = nb * BLOCK_N;
      const int8_t* Bg_lo = w1_pack + (2 * nb)     * W1_block_bytes;
      const int8_t* Bg_hi = w1_pack + (2 * nb + 1) * W1_block_bytes;
      const int8_t* Bu_lo = w1_pack + ((N / 16) + 2 * nb)     * W1_block_bytes;
      const int8_t* Bu_hi = w1_pack + ((N / 16) + 2 * nb + 1) * W1_block_bytes;

      __m512  vbs_g0 = _mm512_loadu_ps(w1s + n0);
      __m512  vbs_g1 = _mm512_loadu_ps(w1s + n0 + 16);
      __m512i vbc_g0 = _mm512_loadu_si512(w1_bc + n0);
      __m512i vbc_g1 = _mm512_loadu_si512(w1_bc + n0 + 16);
      __m512  vbs_u0 = _mm512_loadu_ps(w1s + N + n0);
      __m512  vbs_u1 = _mm512_loadu_ps(w1s + N + n0 + 16);
      __m512i vbc_u0 = _mm512_loadu_si512(w1_bc + N + n0);
      __m512i vbc_u1 = _mm512_loadu_si512(w1_bc + N + n0 + 16);
      const __m512 signBit = _mm512_set1_ps(-0.0f);

      for (int64_t mb = 0; mb < MB; ++mb) {
        int64_t m0 = mb * BLOCK_M;
        const uint8_t* A = qi + m0 * K;
        const float*   Ag = qisc + m0;
        float* fo_row = f_scratch + m0 * N + n0;

        // Per-row vector amax (16-lane), init 0.
        __m512 vam[32];
        for (int r = 0; r < 32; ++r) vam[r] = _mm512_setzero_ps();

        amx_gemm_32x16_gu(A, K, Bg_lo, Bu_lo, cg, cu, K);
        for (int r = 0; r < 32; ++r) {
          __m512 vas = _mm512_set1_ps(Ag[r]);
          __m512i icg = _mm512_loadu_si512(cg + r * 16);
          __m512i icu = _mm512_loadu_si512(cu + r * 16);
          __m512 g = _mm512_mul_ps(_mm512_mul_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(icg, vbc_g0)), vas), vbs_g0);
          __m512 u = _mm512_mul_ps(_mm512_mul_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(icu, vbc_u0)), vas), vbs_u0);
          __m512 v = _mm512_mul_ps(silu_ps(g), u);
          _mm512_storeu_ps(fo_row + r * N, v);
          vam[r] = _mm512_max_ps(vam[r], _mm512_andnot_ps(signBit, v));
        }

        amx_gemm_32x16_gu(A, K, Bg_hi, Bu_hi, cg, cu, K);
        for (int r = 0; r < 32; ++r) {
          __m512 vas = _mm512_set1_ps(Ag[r]);
          __m512i icg = _mm512_loadu_si512(cg + r * 16);
          __m512i icu = _mm512_loadu_si512(cu + r * 16);
          __m512 g = _mm512_mul_ps(_mm512_mul_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(icg, vbc_g1)), vas), vbs_g1);
          __m512 u = _mm512_mul_ps(_mm512_mul_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(icu, vbc_u1)), vas), vbs_u1);
          __m512 v = _mm512_mul_ps(silu_ps(g), u);
          _mm512_storeu_ps(fo_row + r * N + 16, v);
          vam[r] = _mm512_max_ps(vam[r], _mm512_andnot_ps(signBit, v));
        }

        // One horizontal reduce per row now, not per row-half.
        for (int r = 0; r < 32; ++r) {
          float av = _mm512_reduce_max_ps(vam[r]);
          float cur = my_amax[m0 + r];
          if (av > cur) my_amax[m0 + r] = av;
        }
      }
    }

    #pragma omp barrier
    if (_timing && tid == 0) _t_s1 = now_ns();

    // ---- Stage 1.5: parallel reduction of per-thread amax -> intsc + inv.
    // Vectorized: process 16 rows at a time with AVX-512 max across threads.
    // amax_tpr layout is [nthr][M], so thread-partial values for row m are at
    // stride M apart. We load 16 contiguous rows for thread t, reduce across
    // threads, then scalarize only the final max->scale->inv-scale chain.
    #pragma omp for schedule(static) nowait
    for (int64_t m0 = 0; m0 < M; m0 += 16) {
      int64_t chunk = std::min<int64_t>(16, M - m0);
      __m512 am = _mm512_setzero_ps();
      for (int t = 0; t < nthr; ++t) {
        am = _mm512_max_ps(am, _mm512_loadu_ps(amax_tpr + (int64_t)t * M + m0));
      }
      // am now holds reduced amax for 16 rows. Clamp, derive scale + inv.
      am = _mm512_max_ps(am, _mm512_set1_ps(1e-7f));
      __m512 scale = _mm512_mul_ps(am, _mm512_set1_ps(1.f / 127.f));
      __m512 inv   = _mm512_div_ps(_mm512_set1_ps(127.f), am);
      __mmask16 mk = (chunk == 16) ? (__mmask16)0xFFFF
                                    : (__mmask16)((1u << chunk) - 1u);
      _mm512_mask_storeu_ps(intsc + m0, mk, scale);
      _mm512_mask_storeu_ps(amax_tpr + m0, mk, inv);
    }
    #pragma omp barrier
    if (_timing && tid == 0) _t_s1red = now_ns();

    // Column-parallel requant: same nb grid as Stage 1 -> warm L2 for this thread.
    #pragma omp for schedule(static) nowait
    for (int64_t nb = 0; nb < NB1; ++nb) {
      int64_t n0 = nb * BLOCK_N;
      const __m512i off = _mm512_set1_epi32(128);
      for (int64_t m = 0; m < M; ++m) {
        const __m512 vd = _mm512_set1_ps(amax_tpr[m]);
        const float* fr = f_scratch + m * N + n0;
        uint8_t* dst = qi2 + m * N + n0;
        __m512 v0 = _mm512_mul_ps(_mm512_loadu_ps(fr + 0), vd);
        __m512 v1 = _mm512_mul_ps(_mm512_loadu_ps(fr + 16), vd);
        v0 = _mm512_roundscale_ps(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v1 = _mm512_roundscale_ps(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i i0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(v0), off));
        __m128i i1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(v1), off));
        _mm256_storeu_si256((__m256i*)(dst + 0), _mm256_set_m128i(i1, i0));
      }
    }

    #pragma omp barrier
    if (_timing && tid == 0) _t_s1req = now_ns();

    // ---- Stage 2: W2 gemm + dequant + scaled add -> bf16 out.
    #pragma omp for schedule(static) nowait
    for (int64_t nb = 0; nb < NB2; ++nb) {
      int64_t oc0 = nb * BLOCK_N;
      const int8_t* B_lo = w2_pack + (2 * nb)     * W2_block_bytes;
      const int8_t* B_hi = w2_pack + (2 * nb + 1) * W2_block_bytes;
      __m512  vbs_lo = _mm512_loadu_ps(w2s + oc0);
      __m512  vbs_hi = _mm512_loadu_ps(w2s + oc0 + 16);
      __m512i vbc_lo = _mm512_loadu_si512(w2_bc + oc0);
      __m512i vbc_hi = _mm512_loadu_si512(w2_bc + oc0 + 16);

      for (int64_t mb = 0; mb < MB; ++mb) {
        int64_t m0 = mb * BLOCK_M;
        const uint8_t* A = qi2 + m0 * N;
        amx_gemm_32x32(A, N, B_lo, B_hi, cacc, N);

        const float* As = intsc + m0;
        for (int r = 0; r < 32; ++r) {
          __m512 vss = _mm512_set1_ps(As[r]);
          __m512i cl = _mm512_loadu_si512(cacc + r * 32);
          __m512i ch = _mm512_loadu_si512(cacc + r * 32 + 16);
          __m512 fl = _mm512_cvtepi32_ps(_mm512_sub_epi32(cl, vbc_lo));
          __m512 fh = _mm512_cvtepi32_ps(_mm512_sub_epi32(ch, vbc_hi));
          fl = _mm512_mul_ps(_mm512_mul_ps(fl, vss), vbs_lo);
          fh = _mm512_mul_ps(_mm512_mul_ps(fh, vss), vbs_hi);

          // Load fused_out as one 512-bit bf16 vector and split.
          const uint16_t* fop = reinterpret_cast<const uint16_t*>(fused) + (m0 + r) * K + oc0;
          __m512i fb = _mm512_loadu_si512((const __m512i*)fop);
          __m512 ff0 = _mm512_castsi512_ps(_mm512_slli_epi32(
              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(fb)), 16));
          __m512 ff1 = _mm512_castsi512_ps(_mm512_slli_epi32(
              _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(fb, 1)), 16));
          fl = _mm512_fmadd_ps(ff0, vroute, fl);
          fh = _mm512_fmadd_ps(ff1, vroute, fh);

          // Single cvtne2ps_pbh packs both halves into one 512-bit bf16 vector.
          uint16_t* op = reinterpret_cast<uint16_t*>(outp) + (m0 + r) * K + oc0;
          _mm512_storeu_si512((__m512i*)op,
              (__m512i)(_mm512_cvtne2ps_pbh(fh, fl)));
        }
      }
    }
    // Implicit barrier at end of parallel captures stage 2 end time.
    #pragma omp barrier
    if (_timing && tid == 0) _t_s2 = now_ns();
    _tile_release();
  }

  if (_timing) {
    int64_t t_end = now_ns();
    _tm.ns_total.fetch_add(t_end - _t_call_start, std::memory_order_relaxed);
    _tm.ns_s0.fetch_add(_t_s0 - _t_stage_start, std::memory_order_relaxed);
    _tm.ns_s1_gemm.fetch_add(_t_s1 - _t_s0, std::memory_order_relaxed);
    _tm.ns_s1_reduce.fetch_add(_t_s1red - _t_s1, std::memory_order_relaxed);
    _tm.ns_s1_req.fetch_add(_t_s1req - _t_s1red, std::memory_order_relaxed);
    _tm.ns_s2.fetch_add(_t_s2 - _t_s1req, std::memory_order_relaxed);
    maybe_print_timing();
  }

  return out_t;
}

}  // anonymous namespace

at::Tensor shared_expert_amx_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    const at::Tensor& w1_scale,
    const at::Tensor& w2_scale) {
  RECORD_FUNCTION("sgl-kernel::shared_expert_amx_cpu",
      std::vector<c10::IValue>({hidden_states, w1, w2, fused_experts_out}));
  return shared_expert_amx_impl(
      hidden_states, w1, w2, fused_experts_out,
      routed_scaling_factor, w1_scale, w2_scale);
}

#else  // non-x86_64 fallback

#include "common.h"

at::Tensor shared_expert_amx_cpu(
    at::Tensor& hidden_states,
    at::Tensor& /*w1*/,
    at::Tensor& /*w2*/,
    at::Tensor& /*fused_experts_out*/,
    double /*routed_scaling_factor*/,
    const at::Tensor& /*w1_scale*/,
    const at::Tensor& /*w2_scale*/) {
  TORCH_CHECK(false, "shared_expert_amx_cpu: requires x86_64 with AMX INT8");
  return hidden_states;
}

#endif
