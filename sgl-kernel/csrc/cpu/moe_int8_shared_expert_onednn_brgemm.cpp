// moe_int8_shared_expert_onednn_brgemm.cpp
//
// This is a byte-for-byte copy of the baseline INT8 shared-expert kernel
// (moe_int8.cpp :: shared_expert_int8_kernel_impl) with EXACTLY ONE change:
// the two GEMMs are issued through oneDNN's *ukernel* BRGeMM API
// (dnnl::ukernel::brgemm) instead of PyTorch's at::native::cpublas::brgemm.
//
// Everything else -- quantization, VNNI weight packing (convert_weight_packed),
// s8s8 compensation, silu_and_mul, scale_C, add_mul_stub, the parallel_2d /
// loop_2d tiling -- is identical to the baseline. The oneDNN ukernel brgemm is
// a drop-in for cpublas::brgemm because PyTorch's cpublas::brgemm is itself a
// wrapper over oneDNN's brgemm: the VNNI4 packed-B layout [K/4, N, 4] produced
// by convert_weight_packed is precisely what the ukernel brgemm expects for a
// pre-packed B tile, so no re-packing is required.
//
// The public ukernel API is only available when oneDNN is built with
// -DDNNL_EXPERIMENTAL_UKERNEL=ON (the macro is emitted into dnnl_config.h and
// picked up transitively via dnnl.hpp). CMake only compiles this file when that
// macro is present; the #else branch below is a safety stub.

#include "common.h"
#include "gemm.h"
#include "vec.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ukernel.hpp>

#if defined(DNNL_EXPERIMENTAL_UKERNEL)

#include <map>
#include <tuple>
#include <vector>

namespace {

// ===========================================================================
// Drop-in replacement for at::native::cpublas::brgemm (u8 * s8 -> s32, VNNI B).
// Same argument list and semantics; only the backend differs.
//
// PyTorch's cpublas::brgemm is a wrapper over oneDNN's brgemm, so the B tensor
// passed here is already in the VNNI4 layout the ukernel brgemm expects. We
// therefore build a batch_size=1 ukernel over the full K and feed the packed B
// pointer directly (no dnnl::ukernel::transform pass needed).
//
// brgemm objects are cached per (M,N,K,lda,ldb,ldc,add_C) in thread-local
// storage: generate()/finalize() run once per unique config per thread, and
// execute() is called in the hot loop. This mirrors the baseline, where the
// cpublas brgemm kernel is likewise JIT-cached internally.
// ===========================================================================
namespace onednn_brgemm {

using dnnl::ukernel::brgemm;

struct KernelEntry {
  brgemm brg;
  size_t scratchpad_size = 0;
};

using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>;

// Thread-local so kernel generation is race-free and AMX hw-context is per-core.
static thread_local std::map<Key, KernelEntry> g_kernels;
static thread_local std::vector<uint8_t> g_scratchpad;

static KernelEntry& get_kernel(
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc, bool add_C) {
  Key key{M, N, K, lda, ldb, ldc, add_C};
  auto it = g_kernels.find(key);
  if (it != g_kernels.end()) return it->second;

  using dt = dnnl::memory::data_type;
  KernelEntry e;
  // batch_size = 1, whole-K ukernel: a single execute() covers the full K.
  e.brg = brgemm(/*M*/ M, /*N*/ N, /*K*/ K, /*batch_size*/ 1,
                 /*lda*/ lda, /*ldb*/ ldb, /*ldc*/ ldc,
                 /*a_dt*/ dt::u8, /*b_dt*/ dt::s8, /*c_dt*/ dt::s32);
  e.brg.set_add_C(add_C);
  TORCH_CHECK(e.brg.finalize(),
      "onednn_brgemm: ukernel finalize failed for M=", M, " N=", N, " K=", K);
  e.brg.generate();
  e.scratchpad_size = e.brg.get_scratchpad_size();

  auto ins = g_kernels.emplace(std::move(key), std::move(e));
  return ins.first->second;
}

// Signature intentionally mirrors at::native::cpublas::brgemm's u8*s8->s32 form.
static inline void brgemm(
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    bool add_C,
    const uint8_t* A, const int8_t* B, int32_t* C) {
  auto& e = get_kernel(M, N, K, lda, ldb, ldc, add_C);
  if (g_scratchpad.size() < e.scratchpad_size) g_scratchpad.resize(e.scratchpad_size);

  // Prepare AMX tiles for this kernel's config (optimized-out when unchanged).
  e.brg.set_hw_context();

  static const std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>>
      offsets = {{0, 0}};  // batch_size = 1
  e.brg.execute(A, B, offsets, C, g_scratchpad.data());
}

// Drop-in for at::native::cpublas::brgemm_release().
static inline void brgemm_release() {
  brgemm::release_hw_context();
}

}  // namespace onednn_brgemm

// ===========================================================================
// The following helpers are copied verbatim from moe_int8.cpp so this
// translation unit reproduces the baseline shared-expert kernel exactly.
// Keep in sync with moe_int8.cpp if the baseline quant/dequant scheme changes.
// ===========================================================================

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

template <>
inline void copy_stub<uint8_t>(uint8_t* __restrict__ out, const uint8_t* __restrict__ input, int64_t size) {
  std::memcpy(out, input, size * sizeof(uint8_t));
}

template <typename scalar_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const float* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float scale,
    int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_vec = fVec(scale);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec x0 = fVec::loadu(input + d);
    fVec x1 = fVec::loadu(input + d + fVec::size());

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul(
    scalar_t* __restrict__ C,
    const int32_t* __restrict__ C0,  // x: x0, x1
    const int32_t* __restrict__ C1,  // y: y0, y1
    const float* __restrict__ As,
    const float* __restrict__ Bs0,
    const float* __restrict__ Bs1,
    const int32_t* __restrict__ Bcomp0,
    const int32_t* __restrict__ Bcomp1,
    int64_t m_size,
    int64_t N) {
#if defined(CPU_CAPABILITY_AVX512)
  constexpr int COLS = BLOCK_N / 16;
  static_assert(COLS % 2 == 0);

  __m512 vc0[COLS];
  __m512 vc1[COLS];
  __m512i vcomp0[COLS];
  __m512i vcomp1[COLS];
  __m512 vas;
  __m512 vbs0[COLS];
  __m512 vbs1[COLS];

  auto load_scale_and_comp = [&](auto col) {
    vcomp0[col] = _mm512_loadu_si512(Bcomp0 + col * 16);
    vcomp1[col] = _mm512_loadu_si512(Bcomp1 + col * 16);
    vbs0[col] = _mm512_loadu_ps(Bs0 + col * 16);
    vbs1[col] = _mm512_loadu_ps(Bs1 + col * 16);
  };
  Unroll<COLS>{}(load_scale_and_comp);

  auto scalec = [&](auto col, int64_t m) {
    vas = _mm512_set1_ps(As[m]);
    __m512i vc32_0 = _mm512_loadu_si512(C0 + m * BLOCK_N + col * 16);
    __m512i vc32_1 = _mm512_loadu_si512(C1 + m * BLOCK_N + col * 16);
    vc0[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32_0, vcomp0[col]));
    vc1[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32_1, vcomp1[col]));
    vc0[col] = _mm512_mul_ps(_mm512_mul_ps(vc0[col], vas), vbs0[col]);
    vc1[col] = _mm512_mul_ps(_mm512_mul_ps(vc1[col], vas), vbs1[col]);
  };

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  auto silu_and_mul = [&](auto col) {
    fVec x = fVec(vc0[col]);
    fVec y = fVec(vc1[col]);
    x = x / (one + x.neg().exp_u20());
    vc0[col] = x * y;
  };

  auto storec = [&](auto col, int64_t m) {
    if constexpr (col % 2 == 0) {
      fVec x0 = fVec(vc0[col + 0]);
      fVec x1 = fVec(vc0[col + 1]);
      bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
      out_vec.store(C + m * N + col * 16);
    }
  };

  for (int64_t m = 0; m < m_size; ++m) {
    Unroll<COLS>{}(scalec, m);
    Unroll<COLS>{}(silu_and_mul);
    Unroll<COLS>{}(storec, m);
  }
#else
  TORCH_CHECK(false, "silu_and_mul: scalar path not implemented!");
#endif
}

template <int BLOCK_N>
inline void scale_C(
    float* __restrict__ C,
    const int32_t* __restrict__ Ctmp,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    const int32_t* __restrict__ Bcomp,
    int64_t m_size) {
#if defined(CPU_CAPABILITY_AVX512)
  constexpr int COLS = BLOCK_N / 16;
  static_assert(COLS % 2 == 0);

  __m512 vc[COLS];
  __m512i vcomp[COLS];
  __m512 vas;
  __m512 vbs[COLS];

  auto load_scale_and_comp = [&](auto col) {
    vcomp[col] = _mm512_loadu_si512(Bcomp + col * 16);
    vbs[col] = _mm512_loadu_ps(Bs + col * 16);
  };
  Unroll<COLS>{}(load_scale_and_comp);

  auto scalec = [&](auto col, int64_t m) {
    vas = _mm512_set1_ps(As[m]);
    __m512i vc32 = _mm512_loadu_si512(Ctmp + m * BLOCK_N + col * 16);
    vc[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32, vcomp[col]));
    vc[col] = _mm512_mul_ps(_mm512_mul_ps(vc[col], vas), vbs[col]);
    _mm512_storeu_ps(C + m * BLOCK_N + col * 16, vc[col]);
  };

  for (int64_t m = 0; m < m_size; ++m) {
    Unroll<COLS>{}(scalec, m);
  }
#else
  TORCH_CHECK(false, "scale_C: scalar path not implemented!");
#endif
}

// ===========================================================================
// Verbatim copy of moe_int8.cpp :: shared_expert_int8_kernel_impl, with the
// two cpublas::brgemm call sites (+ brgemm_release) swapped for onednn_brgemm.
// The tinygemm (non-brgemm, small-M) fallback is intentionally NOT included:
// this variant always uses the oneDNN brgemm path, matching the intent of
// "call brgemm from oneDNN".
// ===========================================================================
template <typename scalar_t>
void shared_expert_int8_onednn_brgemm_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {
  // handle 2 tiles per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * K, As_tmp[m], input + m * K, K);
    }
  });

  // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  // K and N are packed for int8
  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);
  const int64_t stride_n = packed_K;

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    int32_t* __restrict__ C0 = reinterpret_cast<int32_t*>(C_tmp) + tid * 2 * BLOCK_M * BLOCK_N;
    int32_t* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      // nb_upper from top half and nb_lower from bottom half
      int64_t nb_upper = nb, nb_lower = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);

      // A shape [m_size, K]
      const uint8_t* A = Aq_tmp + mb * BLOCK_M * K;
      const float* As = As_tmp + mb * BLOCK_M;

      // B shape [K, n_size] in vnni format
      const int8_t* __restrict__ B0 = packed_w1 + nb_upper * BLOCK_N * stride_n;
      const int8_t* __restrict__ B1 = packed_w1 + nb_lower * BLOCK_N * stride_n;
      const float* __restrict__ Bs0 = w1s + nb_upper * BLOCK_N;
      const float* __restrict__ Bs1 = w1s + nb_lower * BLOCK_N;

      // 1.b gemm: C0 = A @ B0
      onednn_brgemm::brgemm(
          /* M     */ m_size,
          /* N     */ n_size,
          /* K     */ K,
          /* lda   */ K,
          /* ldb   */ n_size,
          /* ldc   */ BLOCK_N,
          /* add_C */ false,
          /* A     */ A,
          /* B     */ B0,
          /* C     */ C0);

      // 1.c gemm: C1 = A @ B1
      onednn_brgemm::brgemm(
          /* M     */ m_size,
          /* N     */ n_size,
          /* K     */ K,
          /* lda   */ K,
          /* ldb   */ n_size,
          /* ldc   */ BLOCK_N,
          /* add_C */ false,
          /* A     */ A,
          /* B     */ B1,
          /* C     */ C1);

      const int32_t* Bcomp0 = reinterpret_cast<const int32_t*>(B0 + block_size_n() * K);
      const int32_t* Bcomp1 = reinterpret_cast<const int32_t*>(B1 + block_size_n() * K);

      // 1.d silu and mul
      silu_and_mul<scalar_t, BLOCK_N>(
          ic1 + mb * BLOCK_M * N + nb * BLOCK_N, C0, C1, As, Bs0, Bs1, Bcomp0, Bcomp1, m_size, N);
    });

    onednn_brgemm::brgemm_release();
  });

  // stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * N, As_tmp[m], ic1 + m * N, N);
    }
  });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [K, N] as [OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_oc = packed_N;

  // parallel on [MB2, NB2]
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    int32_t* __restrict__ C32 = reinterpret_cast<int32_t*>(C + BLOCK_M * BLOCK_N);

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = std::min(M - mb * BLOCK_M, BLOCK_M);
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A shape [m_size, IC]
      const uint8_t* __restrict__ A = Aq_tmp + mb * BLOCK_M * N;
      const float* __restrict__ As = As_tmp + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      const int8_t* __restrict__ B = packed_w2 + nb * BLOCK_N * stride_oc;
      const float* __restrict__ Bs = w2s + nb * BLOCK_N;

      onednn_brgemm::brgemm(
          /* M     */ m_size,
          /* N     */ n_size,
          /* K     */ IC,
          /* lda   */ IC,
          /* ldb   */ n_size,
          /* ldc   */ BLOCK_N,
          /* add_C */ false,
          /* A     */ A,
          /* B     */ B,
          /* C     */ C32);

      // apply scales
      const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + block_size_n() * IC);
      scale_C<BLOCK_N>(C, C32, As, Bs, Bcomp, m_size);

      // 2.b copy from C to output and add fused_experts_out
      scalar_t* __restrict__ out = output + mb * BLOCK_M * K + nb * BLOCK_N;
      const scalar_t* __restrict__ fused_out = fused_experts_out + mb * BLOCK_M * K + nb * BLOCK_N;
      for (int64_t m = 0; m < m_size; ++m) {
        add_mul_stub(out + m * K, C + m * BLOCK_N, fused_out + m * K, routed_scaling_factor, n_size);
      }
    });

    onednn_brgemm::brgemm_release();
  });
}

}  // anonymous namespace

// Public API -- INT8 shared expert via oneDNN ukernel BRGeMM.
// Layout matches shared_expert_onednn_cpu / shared_expert_amx_cpu:
//   hidden_states: [M, K] bf16, w1: [K, 2N] s8 packed, w2: [N, K] s8 packed.
// This op mirrors shared_expert_cpu's int8 glue: it VNNI-packs the weights via
// convert_weight_packed (skipped when already packed) and dispatches to the
// oneDNN-brgemm variant of the baseline kernel.
at::Tensor shared_expert_onednn_brgemm_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    const at::Tensor& w1_scale,
    const at::Tensor& w2_scale) {
  RECORD_FUNCTION("sgl-kernel::shared_expert_onednn_brgemm_cpu",
      std::vector<c10::IValue>({hidden_states, w1, w2, fused_experts_out}));

  const auto st = hidden_states.scalar_type();
  CHECK_INPUT(hidden_states);
  CHECK_INPUT(fused_experts_out);
  CHECK_INPUT(w1);
  CHECK_INPUT(w2);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, w1);
  CHECK_DIM(2, w2);
  CHECK_EQ(hidden_states.sizes(), fused_experts_out.sizes());
  CHECK_EQ(hidden_states.scalar_type(), st);

  // Weights arrive already VNNI-packed (is_vnni path) from the benchmark, same
  // as shared_expert_cpu's int8 branch. Pack here if not.
  auto packed_w1 = w1.size(1) == get_row_size<int8_t>(hidden_states.size(1)) ? w1 : convert_weight_packed(w1);
  auto packed_w2 = w2.size(1) == get_row_size<int8_t>(w1.size(0) / 2) ? w2 : convert_weight_packed(w2);

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = w1.size(0) / 2;

  int64_t packed_K = get_row_size<int8_t>(K);
  int64_t packed_N = get_row_size<int8_t>(N);
  CHECK_EQ(w2.size(0), K);
  CHECK_EQ(packed_w1.size(1), packed_K);
  CHECK_EQ(packed_w2.size(1), packed_N);

  auto w1s = w1_scale.scalar_type() == at::kFloat ? w1_scale : w1_scale.to(at::kFloat);
  auto w2s = w2_scale.scalar_type() == at::kFloat ? w2_scale : w2_scale.to(at::kFloat);
  TORCH_CHECK(w1s.numel() == 2 * N);
  TORCH_CHECK(w2s.numel() == K);

  auto out_hidden_states = at::empty_like(hidden_states);

  // Scratch layout mirrors shared_expert_cpu's int8 branch.
  int num_threads = at::get_num_threads();
  int64_t buffer_size_nbytes = M * N * 2 + num_threads * 2 * BLOCK_M * BLOCK_N * sizeof(float);
  buffer_size_nbytes += std::max(M * K, M * N) + M * sizeof(float);
  auto buffer = at::empty({buffer_size_nbytes}, hidden_states.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "shared_expert_onednn_brgemm_kernel_impl", [&] {
    scalar_t* __restrict__ intermediate_cache1 = (scalar_t*)((void*)(buffer.data_ptr<int8_t>()));
    float* __restrict__ C_tmp = (float*)((void*)(intermediate_cache1 + M * N));
    uint8_t* __restrict__ Aq_tmp = (uint8_t*)((void*)(C_tmp + num_threads * 2 * BLOCK_M * BLOCK_N));
    float* __restrict__ As_tmp = (float*)((void*)(Aq_tmp + std::max(M * K, M * N)));

    shared_expert_int8_onednn_brgemm_kernel_impl<scalar_t>(
        out_hidden_states.data_ptr<scalar_t>(),
        intermediate_cache1,
        C_tmp,
        Aq_tmp,
        As_tmp,
        hidden_states.data_ptr<scalar_t>(),
        packed_w1.data_ptr<int8_t>(),
        packed_w2.data_ptr<int8_t>(),
        w1s.data_ptr<float>(),
        w2s.data_ptr<float>(),
        fused_experts_out.data_ptr<scalar_t>(),
        (float)routed_scaling_factor,
        M,
        N,
        K);
  });

  return out_hidden_states;
}

#else  // !DNNL_EXPERIMENTAL_UKERNEL

#include "common.h"

at::Tensor shared_expert_onednn_brgemm_cpu(
    at::Tensor& hidden_states,
    at::Tensor& /*w1*/,
    at::Tensor& /*w2*/,
    at::Tensor& /*fused_experts_out*/,
    double /*routed_scaling_factor*/,
    const at::Tensor& /*w1_scale*/,
    const at::Tensor& /*w2_scale*/) {
  TORCH_CHECK(false,
      "shared_expert_onednn_brgemm_cpu: oneDNN was not built with "
      "DNNL_EXPERIMENTAL_UKERNEL=ON; the ukernel BRGeMM API is unavailable");
  return hidden_states;
}

#endif  // DNNL_EXPERIMENTAL_UKERNEL
