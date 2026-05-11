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

// shared_expert_onednn.cpp -- Shared-expert INT8 (u8*s8) using oneDNN matmul.
// Uses bf16 matmul dst + fused per-col wei-scale (the "lite" path): cuts half
// the dst memory traffic vs f32/s32 dst and keeps the int8→bf16 JIT epilog.

#include <list>
#include <utility>

#include <oneapi/dnnl/dnnl.hpp>

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

static inline float fast_expf(float x) {
  const float log2e = 1.4426950408889634f;
  float t = x * log2e;
  t = t < -126.f ? -126.f : (t > 126.f ? 126.f : t);
  float ti = __builtin_floorf(t), tf = t - ti;
  int32_t n = (int32_t)ti;
  float p = 1.f + tf * (0.6931472f + tf * (0.2402265f + tf * (0.0555041f +
             tf * (0.00961813f + tf * 0.00134326f))));
  union { float f; int32_t i; } s;
  s.i = (n + 127) << 23;
  return p * s.f;
}

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
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
#define _CVT_BF16_TO_FP32(a) \
    _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16))
#endif

// Quantize BF16 -> u8 with +128 offset (row-parallel, AVX-512 fast path).
static void quantize_bf16_to_u8(
    const at::BFloat16* src, uint8_t* dst, float* sc,
    int64_t M, int64_t K) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512 signBit = _mm512_set1_ps(-0.0f);
  const __m512i off = _mm512_set1_epi32(128);
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; m++) {
    const at::BFloat16* r = src + m * K;
    uint8_t* o = dst + m * K;

    __m512 vamax0 = _mm512_setzero_ps();
    __m512 vamax1 = _mm512_setzero_ps();
    int64_t k = 0;
    for (; k + 32 <= K; k += 32) {
      __m512i va = _mm512_loadu_si512((void*)(r + k));
      __m512 va0 = _CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
      __m512 va1 = _CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
      vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, va0));
      vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(signBit, va1));
    }
    float amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
    for (; k < K; ++k) amax = std::max(amax, std::abs((float)r[k]));
    amax = std::max(amax, 1e-7f);
    sc[m] = amax / 127.f;
    const float inv = 127.f / amax;
    const __m512 vd = _mm512_set1_ps(inv);

    k = 0;
    for (; k + 32 <= K; k += 32) {
      __m512i va = _mm512_loadu_si512((void*)(r + k));
      __m512 va0 = _CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
      __m512 va1 = _CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
      va0 = _mm512_mul_ps(va0, vd);
      va1 = _mm512_mul_ps(va1, vd);
      va0 = _mm512_roundscale_ps(va0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      va1 = _mm512_roundscale_ps(va1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      __m128i i0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va0), off));
      __m128i i1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va1), off));
      _mm256_storeu_si256((__m256i*)(o + k), _mm256_set_m128i(i1, i0));
    }
    for (; k < K; ++k) {
      o[k] = (uint8_t)(std::round((float)r[k] * inv)) + 128;
    }
  }
#else
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; m++) {
    const at::BFloat16* r = src + m * K;
    uint8_t* o = dst + m * K;
    float amax = 0;
    for (int64_t k = 0; k < K; k++)
      amax = std::max(amax, std::abs((float)r[k]));
    amax = std::max(amax, 1e-7f);
    float scale = amax / 127.f, inv = 127.f / amax;
    sc[m] = scale;
    for (int64_t k = 0; k < K; k++)
      o[k] = (uint8_t)((int32_t)std::round((float)r[k] * inv) + 128);
  }
#endif
}

// Stage 1b post: (s32 - Bcomp) * As * Bs -> SiLU*Mul -> u8
// Pass A: compute SiLU(G)*U into f32 scratch while tracking exact amax.
// Pass B: stream scratch -> scale by inv -> u8. No second exp/div pass.
// At N=2048, per-row scratch is 8KB (fits in L1), so the scratch round-trip
// is cheaper than recomputing SiLU*Mul.
// Using s32 matmul dst (vs bf16) preserves full accumulator precision through
// the Bcomp subtract — bf16 dst loses ~0.5 per element after dequant, which
// destroys accuracy once Bcomp (~100K scale) is subtracted.
static void dequant_silu_mul_to_u8(
    const int32_t* gemm_out,   // [M, 2*N] s32
    uint8_t* qi2,              // [M, N] u8 out
    float* intsc,              // [M] per-row scales out
    float* scratch,            // [M, N] f32 scratch (thread-partitioned by row)
    const float* src_sc,       // [M] input per-row scales
    const float* w1_scales,    // [2*N] (gate at [0..N), up at [N..2N))
    const int32_t* gate_bcomp, // [N]
    const int32_t* up_bcomp,   // [N]
    int64_t M, int64_t N) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512 signBit = _mm512_set1_ps(-0.0f);
  const __m512i off = _mm512_set1_epi32(128);
#endif
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < M; ++i) {
    float ss = src_sc[i];
    const int32_t* row = gemm_out + i * 2 * N;
    const float* gws = w1_scales;
    const float* uws = w1_scales + N;
    uint8_t* o = qi2 + i * N;
    float* sc_row = scratch + i * N;

    // ---- Pass A: compute SiLU(G)*U, write f32 scratch, track amax ----
    float amax = 0.f;
    int64_t n = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 vss = _mm512_set1_ps(ss);
    __m512 v1 = _mm512_set1_ps(1.f);
    __m512 vamax0 = _mm512_setzero_ps(), vamax1 = _mm512_setzero_ps();
    for (; n + 32 <= N; n += 32) {
      __m512 g0 = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + n), _mm512_loadu_si512(gate_bcomp + n))),
          vss), _mm512_loadu_ps(gws + n));
      __m512 u0 = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + N + n), _mm512_loadu_si512(up_bcomp + n))),
          vss), _mm512_loadu_ps(uws + n));
      __m512 sig0 = _mm512_div_ps(v1,
          _mm512_add_ps(v1, _mm512_fast_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g0))));
      __m512 v0 = _mm512_mul_ps(_mm512_mul_ps(g0, sig0), u0);

      __m512 g1 = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + n + 16), _mm512_loadu_si512(gate_bcomp + n + 16))),
          vss), _mm512_loadu_ps(gws + n + 16));
      __m512 u1 = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + N + n + 16), _mm512_loadu_si512(up_bcomp + n + 16))),
          vss), _mm512_loadu_ps(uws + n + 16));
      __m512 sig1 = _mm512_div_ps(v1,
          _mm512_add_ps(v1, _mm512_fast_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g1))));
      __m512 v1x = _mm512_mul_ps(_mm512_mul_ps(g1, sig1), u1);

      _mm512_storeu_ps(sc_row + n, v0);
      _mm512_storeu_ps(sc_row + n + 16, v1x);
      vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, v0));
      vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(signBit, v1x));
    }
    for (; n + 16 <= N; n += 16) {
      __m512 g = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + n), _mm512_loadu_si512(gate_bcomp + n))),
          vss), _mm512_loadu_ps(gws + n));
      __m512 u = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + N + n), _mm512_loadu_si512(up_bcomp + n))),
          vss), _mm512_loadu_ps(uws + n));
      __m512 sig = _mm512_div_ps(v1,
          _mm512_add_ps(v1, _mm512_fast_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g))));
      __m512 v = _mm512_mul_ps(_mm512_mul_ps(g, sig), u);
      _mm512_storeu_ps(sc_row + n, v);
      vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, v));
    }
    amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
#endif
    for (; n < N; ++n) {
      float g = ss * gws[n] * (float)(row[n] - gate_bcomp[n]);
      float u = ss * uws[n] * (float)(row[N + n] - up_bcomp[n]);
      float silu = g / (1.f + fast_expf(-g));
      float v = silu * u;
      sc_row[n] = v;
      float av = std::abs(v);
      if (av > amax) amax = av;
    }

    amax = std::max(amax, 1e-7f);
    intsc[i] = amax / 127.f;
    const float inv = 127.f / amax;

    // ---- Pass B: stream scratch -> scale -> u8 quantize ----
    n = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512 vd = _mm512_set1_ps(inv);
    for (; n + 32 <= N; n += 32) {
      __m512 v0 = _mm512_mul_ps(_mm512_loadu_ps(sc_row + n), vd);
      v0 = _mm512_roundscale_ps(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      __m512 v1x = _mm512_mul_ps(_mm512_loadu_ps(sc_row + n + 16), vd);
      v1x = _mm512_roundscale_ps(v1x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      __m128i i0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(v0), off));
      __m128i i1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(v1x), off));
      _mm256_storeu_si256((__m256i*)(o + n), _mm256_set_m128i(i1, i0));
    }
#endif
    for (; n < N; ++n) {
      o[n] = (uint8_t)(std::round(sc_row[n] * inv)) + 128;
    }
  }
}

// Stage 2b post: (s32 - Bcomp) * As * Bs + routed_scaling * fused_out -> bf16
static void dequant_add_scaled_to_bf16(
    const int32_t* gemm_out,      // [M, K] s32
    at::BFloat16* output,         // [M, K] bf16
    const at::BFloat16* fused_out,// [M, K] bf16
    float routed_scaling,
    const float* src_sc,          // [M]
    const float* wsc,             // [K] per-col wei scales
    const int32_t* bcomp,         // [K]
    int64_t M, int64_t K) {
  uint16_t* out = reinterpret_cast<uint16_t*>(output);
  const uint16_t* fo = reinterpret_cast<const uint16_t*>(fused_out);
  const int nth = omp_get_max_threads();
  int64_t tiles_per_row = std::max<int64_t>(1, (int64_t)nth / std::max<int64_t>(M, 1));
  int64_t tile_K = ((K / tiles_per_row) + 15) / 16 * 16;
  if (tile_K < 64) tile_K = 64;
  const int64_t nk_tiles = (K + tile_K - 1) / tile_K;
  const int64_t n_tiles = M * nk_tiles;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512 vscale = _mm512_set1_ps(routed_scaling);
#endif
  #pragma omp parallel for schedule(static)
  for (int64_t it = 0; it < n_tiles; ++it) {
    int64_t i = it / nk_tiles;
    int64_t kt = it % nk_tiles;
    int64_t k0 = kt * tile_K;
    int64_t k1 = std::min(k0 + tile_K, K);
    float ss = src_sc[i];
    const int32_t* row = gemm_out + i * K;
    uint16_t* o = out + i * K;
    const uint16_t* foi = fo + i * K;
    int64_t k = k0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 vss = _mm512_set1_ps(ss);
    for (; k + 16 <= k1; k += 16) {
      __m512 val = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + k), _mm512_loadu_si512(bcomp + k))),
          vss), _mm512_loadu_ps(wsc + k));
      __m256i fb = _mm256_loadu_si256((const __m256i*)(foi + k));
      __m512 ff = _mm512_castsi512_ps(_mm512_slli_epi32(
          _mm512_cvtepu16_epi32(fb), 16));
      val = _mm512_fmadd_ps(ff, vscale, val);
      __m512i ri = _mm512_castps_si512(val);
      __m512i bias = _mm512_and_si512(_mm512_srli_epi32(ri, 16), _mm512_set1_epi32(1));
      ri = _mm512_add_epi32(_mm512_add_epi32(ri, _mm512_set1_epi32(0x7FFF)), bias);
      _mm256_storeu_si256((__m256i*)(o + k), _mm512_cvtepi32_epi16(_mm512_srli_epi32(ri, 16)));
    }
#endif
    for (; k < k1; ++k) {
      float val = ss * wsc[k] * (float)(row[k] - bcomp[k]);
      uint16_t fb = foi[k];
      uint32_t fub = ((uint32_t)fb) << 16;
      float ff; std::memcpy(&ff, &fub, 4);
      val += ff * routed_scaling;
      uint32_t rb; std::memcpy(&rb, &val, 4);
      o[k] = (uint16_t)((rb + 0x7FFF + ((rb >> 16) & 1)) >> 16);
    }
  }
}

// Per-weight-pair cache entry: reordered weights + s32 Bcomp.
// Populated on first call with a new (w1_ptr, w2_ptr) pair, reused afterwards
// for zero-reorder-cost steady state.
// Wei scales are NOT fused into the matmul (s32 dst keeps full precision);
// the post-kernel applies them on the fly.
struct WeightCacheEntry {
  dnnl::memory w1_mem, w2_mem;
  at::Tensor gate_bcomp_t, up_bcomp_t, w2_bcomp_t;
};

// Shape-scoped context holds primitives + scratchpads + pooled intermediate
// buffers + pre-built dnnl::memory wrappers. All of these depend only on MNK,
// so allocating them once per shape removes ~5-20us of per-call overhead that
// dominates at small M.
struct SharedExpertOneDNNContext {
  dnnl::engine eng;
  dnnl::stream strm;
  dnnl::matmul mm1, mm2;
  dnnl::matmul::primitive_desc pd1, pd2;
  dnnl::memory sp1, sp2;

  // Pooled intermediate buffers.
  at::Tensor qi_t, isc_t, intsc_t, qi2_t, d1_t, d2_t, ic1_f32_t;
  // Pre-built dnnl::memory wrappers pointing at the pooled buffers.
  dnnl::memory s1m, d1m, s2m, d2m;

  using WeightKey = std::pair<const void*, const void*>;
  struct KeyHash {
    size_t operator()(const WeightKey& k) const noexcept {
      return std::hash<const void*>{}(k.first) ^
             (std::hash<const void*>{}(k.second) << 1);
    }
  };
  // Bounded to avoid unbounded memory growth if callers churn weights.
  // Capacity ~= typical L in deep MoE models; LRU on overflow.
  std::unordered_map<WeightKey, WeightCacheEntry, KeyHash> wcache;
  std::list<WeightKey> lru;
  std::unordered_map<WeightKey, std::list<WeightKey>::iterator, KeyHash> lru_it;
  static constexpr size_t kMaxCached = 128;

  int64_t M = -1, K = -1, N = -1;
  bool initialized = false;
};

static SharedExpertOneDNNContext& get_shared_expert_context() {
  static SharedExpertOneDNNContext ctx;
  return ctx;
}

// (Re)build primitives when shape changes. Invalidates weight cache because
// the packed weights_desc() depends on the primitive's chosen format.
static void ensure_shape_context(int64_t M, int64_t K, int64_t N) {
  auto& ctx = get_shared_expert_context();
  if (ctx.initialized && ctx.M == M && ctx.K == K && ctx.N == N) {
    return;
  }

  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;

  if (!ctx.initialized) {
    ctx.eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
    ctx.strm = dnnl::stream(ctx.eng);
  }

  // s32 dst preserves full accumulator precision; post-kernel dequants in f32.
  // bf16 dst would round the matmul output to ~3 significant digits, and that
  // rounding dominates error once Bcomp (O(100K)) is subtracted at K=7168.
  dnnl::primitive_attr attr1;
  attr1.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto s1md = dnnl::memory::desc({M, K}, dt::u8, tag::ab);
  auto w1md = dnnl::memory::desc({K, 2 * N}, dt::s8, tag::any);
  auto d1md = dnnl::memory::desc({M, 2 * N}, dt::s32, tag::ab);
  ctx.pd1 = dnnl::matmul::primitive_desc(ctx.eng, s1md, w1md, d1md, attr1);
  ctx.mm1 = dnnl::matmul(ctx.pd1);
  ctx.sp1 = dnnl::memory(ctx.pd1.scratchpad_desc(), ctx.eng);

  dnnl::primitive_attr attr2;
  attr2.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto s2md = dnnl::memory::desc({M, N}, dt::u8, tag::ab);
  auto w2md = dnnl::memory::desc({N, K}, dt::s8, tag::any);
  auto d2md = dnnl::memory::desc({M, K}, dt::s32, tag::ab);
  ctx.pd2 = dnnl::matmul::primitive_desc(ctx.eng, s2md, w2md, d2md, attr2);
  ctx.mm2 = dnnl::matmul(ctx.pd2);
  ctx.sp2 = dnnl::memory(ctx.pd2.scratchpad_desc(), ctx.eng);

  // Pool intermediate buffers: allocated once per shape, reused across calls.
  ctx.qi_t    = at::empty({M, K}, at::kByte);
  ctx.isc_t   = at::empty({M}, at::kFloat);
  ctx.intsc_t = at::empty({M}, at::kFloat);
  ctx.qi2_t   = at::empty({M, N}, at::kByte);
  ctx.d1_t    = at::empty({M, 2 * N}, at::kInt);
  ctx.d2_t    = at::empty({M, K}, at::kInt);
  // f32 scratch for stage-1b pass A -> pass B fusion (per-row L1-resident at N=2048).
  ctx.ic1_f32_t = at::empty({M, N}, at::kFloat);
  // Pre-build dnnl::memory wrappers pointing at the pooled buffers.
  ctx.s1m = dnnl::memory(s1md, ctx.eng, ctx.qi_t.data_ptr<uint8_t>());
  ctx.d1m = dnnl::memory(d1md, ctx.eng, ctx.d1_t.data_ptr<int32_t>());
  ctx.s2m = dnnl::memory(s2md, ctx.eng, ctx.qi2_t.data_ptr<uint8_t>());
  ctx.d2m = dnnl::memory(d2md, ctx.eng, ctx.d2_t.data_ptr<int32_t>());

  // Shape changed: previously cached weights use the old weights_desc() and
  // are now invalid.
  ctx.wcache.clear();
  ctx.lru.clear();
  ctx.lru_it.clear();

  ctx.M = M; ctx.K = K; ctx.N = N;
  ctx.initialized = true;
}

// Returns a cache entry for (w1_data, w2_data), reordering weights and
// precomputing Bcomp on first sight. Subsequent calls return the cached entry
// in O(1). LRU-evicts when the cache reaches capacity.
static WeightCacheEntry& get_or_build_weight_entry(
    const int8_t* w1_data, const int8_t* w2_data,
    const float* w1s, const float* w2s,
    int64_t K, int64_t N) {
  auto& ctx = get_shared_expert_context();
  SharedExpertOneDNNContext::WeightKey key{w1_data, w2_data};

  auto it = ctx.wcache.find(key);
  if (it != ctx.wcache.end()) {
    // LRU touch
    ctx.lru.erase(ctx.lru_it[key]);
    ctx.lru.push_front(key);
    ctx.lru_it[key] = ctx.lru.begin();
    return it->second;
  }

  // Evict LRU if at capacity.
  if (ctx.wcache.size() >= SharedExpertOneDNNContext::kMaxCached) {
    auto victim = ctx.lru.back();
    ctx.lru.pop_back();
    ctx.lru_it.erase(victim);
    ctx.wcache.erase(victim);
  }

  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;
  WeightCacheEntry e;

  auto w1u = dnnl::memory(
      dnnl::memory::desc({K, 2 * N}, dt::s8, tag::ab),
      ctx.eng, const_cast<int8_t*>(w1_data));
  e.w1_mem = dnnl::memory(ctx.pd1.weights_desc(), ctx.eng);
  dnnl::reorder(w1u, e.w1_mem).execute(ctx.strm, w1u, e.w1_mem);

  auto w2u = dnnl::memory(
      dnnl::memory::desc({N, K}, dt::s8, tag::ab),
      ctx.eng, const_cast<int8_t*>(w2_data));
  e.w2_mem = dnnl::memory(ctx.pd2.weights_desc(), ctx.eng);
  dnnl::reorder(w2u, e.w2_mem).execute(ctx.strm, w2u, e.w2_mem);
  ctx.strm.wait();

  // s32 Bcomp: 128 * sum_k(w[k,n]). Post-kernel does (s32_out - Bcomp) * ss * wsc.
  e.gate_bcomp_t = at::empty({N}, at::kInt);
  e.up_bcomp_t   = at::empty({N}, at::kInt);
  e.w2_bcomp_t   = at::empty({K}, at::kInt);
  int32_t* gbc = e.gate_bcomp_t.data_ptr<int32_t>();
  int32_t* ubc = e.up_bcomp_t.data_ptr<int32_t>();
  int32_t* w2bc = e.w2_bcomp_t.data_ptr<int32_t>();
  #pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; n++) {
    int32_t gs = 0, us = 0;
    for (int64_t k = 0; k < K; k++) {
      gs += w1_data[k * 2 * N + n];
      us += w1_data[k * 2 * N + N + n];
    }
    gbc[n] = 128 * gs;
    ubc[n] = 128 * us;
  }
  #pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < K; k++) {
    int32_t s = 0;
    for (int64_t n = 0; n < N; n++) s += w2_data[n * K + k];
    w2bc[k] = 128 * s;
  }

  auto ins = ctx.wcache.emplace(key, std::move(e));
  ctx.lru.push_front(key);
  ctx.lru_it[key] = ctx.lru.begin();
  return ins.first->second;
}

}  // anonymous namespace

// Public API -- INT8 shared expert via oneDNN matmul.
//   hidden_states:     [M, K] bf16
//   w1:                [K, 2*N] s8 (note: transposed vs BRGEMM layout [2*N, K])
//   w2:                [N, K]   s8 (note: transposed vs BRGEMM layout [K, N])
//   fused_experts_out: [M, K] bf16
//   w1_scale:          [2*N] f32
//   w2_scale:          [K]   f32
// Returns bf16 [M, K]: SiLU(xW1_gate) * (xW1_up) * W2 + fused_experts_out * routed_scaling.
at::Tensor shared_expert_onednn_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    const at::Tensor& w1_scale,
    const at::Tensor& w2_scale) {
  RECORD_FUNCTION("sgl-kernel::shared_expert_onednn_cpu",
      std::vector<c10::IValue>({hidden_states, w1, w2, fused_experts_out}));

  TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16,
      "shared_expert_onednn_cpu: only BFloat16 input supported");
  TORCH_CHECK(fused_experts_out.scalar_type() == at::kBFloat16,
      "shared_expert_onednn_cpu: fused_experts_out must be BFloat16");
  TORCH_CHECK(w1.scalar_type() == at::kChar, "w1 must be int8");
  TORCH_CHECK(w2.scalar_type() == at::kChar, "w2 must be int8");

  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, w1);
  CHECK_DIM(2, w2);
  CHECK_DIM(2, fused_experts_out);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = w1.size(1) / 2;

  CHECK_EQ(w1.size(0), K);
  CHECK_EQ(w1.size(1), 2 * N);
  CHECK_EQ(w2.size(0), N);
  CHECK_EQ(w2.size(1), K);
  CHECK_EQ(fused_experts_out.size(0), M);
  CHECK_EQ(fused_experts_out.size(1), K);
  CHECK_EQ(w1_scale.numel(), 2 * N);
  CHECK_EQ(w2_scale.numel(), K);

  auto w1_scale_f = w1_scale.scalar_type() == at::kFloat ? w1_scale : w1_scale.to(at::kFloat);
  auto w2_scale_f = w2_scale.scalar_type() == at::kFloat ? w2_scale : w2_scale.to(at::kFloat);
  float* w1s = w1_scale_f.data_ptr<float>();
  float* w2s = w2_scale_f.data_ptr<float>();
  const int8_t* w1_data = w1.data_ptr<int8_t>();
  const int8_t* w2_data = w2.data_ptr<int8_t>();

  // Primitives (shape-only) are built once per unique MNK. Reordered weights
  // + pre-scaled Bcomp are cached per (w1_ptr, w2_ptr), so repeated calls on
  // the same tensors (e.g. static per-layer weights in a 58-layer loop) pay
  // the reorder cost only on first sight.
  ensure_shape_context(M, K, N);
  auto& ctx = get_shared_expert_context();
  auto& we = get_or_build_weight_entry(w1_data, w2_data, w1s, w2s, K, N);

  auto* inp = hidden_states.data_ptr<at::BFloat16>();
  auto* fused = fused_experts_out.data_ptr<at::BFloat16>();

  auto out_t = at::empty({M, K}, at::kBFloat16);

  // Pull pooled buffer pointers from the shape-scoped context.
  uint8_t* qi     = ctx.qi_t.data_ptr<uint8_t>();
  float*   isc    = ctx.isc_t.data_ptr<float>();
  float*   intsc  = ctx.intsc_t.data_ptr<float>();
  uint8_t* qi2    = ctx.qi2_t.data_ptr<uint8_t>();
  int32_t* d1     = ctx.d1_t.data_ptr<int32_t>();
  int32_t* d2     = ctx.d2_t.data_ptr<int32_t>();
  at::BFloat16* outp = out_t.data_ptr<at::BFloat16>();

  // Raw oneDNN C-API handles for execute: bypasses the per-call
  // std::unordered_map<int, dnnl::memory> construction.
  dnnl_stream_t raw_stream = ctx.strm.get();
  dnnl_memory_t raw_s1 = ctx.s1m.get(), raw_d1 = ctx.d1m.get();
  dnnl_memory_t raw_s2 = ctx.s2m.get(), raw_d2 = ctx.d2m.get();
  dnnl_memory_t raw_sp1 = ctx.sp1.get(), raw_sp2 = ctx.sp2.get();
  dnnl_memory_t raw_w1 = we.w1_mem.get(), raw_w2 = we.w2_mem.get();
  dnnl_exec_arg_t args1[] = {
      {DNNL_ARG_SRC, raw_s1}, {DNNL_ARG_WEIGHTS, raw_w1},
      {DNNL_ARG_DST, raw_d1}, {DNNL_ARG_SCRATCHPAD, raw_sp1}};
  dnnl_exec_arg_t args2[] = {
      {DNNL_ARG_SRC, raw_s2}, {DNNL_ARG_WEIGHTS, raw_w2},
      {DNNL_ARG_DST, raw_d2}, {DNNL_ARG_SCRATCHPAD, raw_sp2}};

  // Stage 0: quantize input bf16 -> u8
  quantize_bf16_to_u8(inp, qi, isc, M, K);

  // Stage 1a: matmul w1 (u8 * s8 -> s32)
  dnnl_primitive_execute(ctx.mm1.get(), raw_stream,
      sizeof(args1) / sizeof(args1[0]), args1);
  dnnl_stream_wait(raw_stream);

  // Stage 1b: dequant + SiLU*Mul + u8 quantize -> qi2
  dequant_silu_mul_to_u8(
      d1, qi2, intsc, ctx.ic1_f32_t.data_ptr<float>(), isc, w1s,
      we.gate_bcomp_t.data_ptr<int32_t>(),
      we.up_bcomp_t.data_ptr<int32_t>(),
      M, N);

  // Stage 2a: matmul w2 (u8 * s8 -> s32)
  dnnl_primitive_execute(ctx.mm2.get(), raw_stream,
      sizeof(args2) / sizeof(args2[0]), args2);
  dnnl_stream_wait(raw_stream);

  // Stage 2b: dequant + scaled fused_out add -> bf16 out
  dequant_add_scaled_to_bf16(
      d2, outp, fused, (float)routed_scaling_factor,
      intsc, w2s, we.w2_bcomp_t.data_ptr<int32_t>(),
      M, K);

  return out_t;
}
