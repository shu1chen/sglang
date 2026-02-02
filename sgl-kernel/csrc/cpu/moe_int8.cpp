#include "common.h"
#include "gemm.h"
#include "vec.h"

#ifdef SGLANG_USE_ONEDNN_MOE
#include "oneapi/dnnl/dnnl.hpp"
#include <vector>
#include <unordered_map>
#include <set>
#include <mutex>

// Unpack VNNI-packed int8 weights to [N, K] format
// VNNI packing within a block: packed[k_group * BLOCK_N * 4 + n * 4 + d] = weight[n][k_group * 4 + d]
// This function unpacks to [N, K] format (row-major layout)
// Transpose is achieved through oneDNN memory strides when needed
// Note: Blocks are stored sequentially, each block occupies BLOCK_N * (K + 4) bytes:
//   - BLOCK_N * K bytes for data in VNNI format
//   - BLOCK_N * 4 bytes for compensation (stored after the data)
template <int BLOCK_N>
inline void unpack_vnni_int8_no_transpose(
    int8_t* __restrict__ unpacked,
    const int8_t* __restrict__ packed,
    int64_t N,  // output rows
    int64_t K) {  // output columns
  constexpr int VNNI_BLK = 4;
  // Each block: BLOCK_N * K bytes data + BLOCK_N * 4 bytes compensation
  const int64_t packed_block_size = BLOCK_N * (K + 4);
  
  // Process each BLOCK_N chunk
  for (int64_t nb = 0; nb < (N + BLOCK_N - 1) / BLOCK_N; ++nb) {
    const int64_t n_start = nb * BLOCK_N;
    const int64_t n_end = std::min(n_start + BLOCK_N, N);
    const int64_t block_n = n_end - n_start;
    
    // Pointer to this block's data (skip previous blocks)
    const int8_t* packed_block = packed + nb * packed_block_size;
    
    // Unpack: output[n][k] = output[n * K + k]
    // Within a block: packed[k_group * BLOCK_N * 4 + n * 4 + d] contains weight[n][k_group*4+d]
    for (int64_t n = 0; n < block_n; ++n) {
      for (int64_t k_group = 0; k_group < K / VNNI_BLK; ++k_group) {
        for (int64_t d = 0; d < VNNI_BLK; ++d) {
          int64_t src_idx = k_group * BLOCK_N * VNNI_BLK + n * VNNI_BLK + d;
          int64_t k = k_group * VNNI_BLK + d;
          int64_t n_global = n_start + n;
          // No transpose: [n_global, k] = [n_global * K + k]
          unpacked[n_global * K + k] = packed_block[src_idx];
        }
      }
    }
  }
}

// apply scales and silu_and_mul in one pass
// Fuses: C_scaled = As * (C - Bcomp) * Bs, then silu(C0_scaled) * C1_scaled
// Processes int32 matmul output directly without intermediate float buffer
template <typename scalar_t>
inline void scale_and_silu_mul_fused(
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ C0,  // gate output (int32)
    const int32_t* __restrict__ C1,  // up output (int32)
    float a_scale,
    const float* __restrict__ Bs0,   // gate weight scales
    const float* __restrict__ Bs1,   // up weight scales
    const int32_t* __restrict__ Bcomp0,  // gate compensation
    const int32_t* __restrict__ Bcomp1,  // up compensation
    int64_t N) {
#if defined(CPU_CAPABILITY_AVX512)
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();  // 32 for bf16
  const fVec one = fVec(1.f);
  const fVec vas = fVec(a_scale);
  
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= N - kVecSize; d += kVecSize) {
    // Load int32 outputs
    __m512i c0_i32_0 = _mm512_loadu_si512(C0 + d);
    __m512i c0_i32_1 = _mm512_loadu_si512(C0 + d + 16);
    __m512i c1_i32_0 = _mm512_loadu_si512(C1 + d);
    __m512i c1_i32_1 = _mm512_loadu_si512(C1 + d + 16);
    
    // Load compensation
    __m512i comp0_0 = _mm512_loadu_si512(Bcomp0 + d);
    __m512i comp0_1 = _mm512_loadu_si512(Bcomp0 + d + 16);
    __m512i comp1_0 = _mm512_loadu_si512(Bcomp1 + d);
    __m512i comp1_1 = _mm512_loadu_si512(Bcomp1 + d + 16);
    
    // Subtract compensation and convert to float
    fVec x0 = fVec(_mm512_cvtepi32_ps(_mm512_sub_epi32(c0_i32_0, comp0_0)));
    fVec x1 = fVec(_mm512_cvtepi32_ps(_mm512_sub_epi32(c0_i32_1, comp0_1)));
    fVec y0 = fVec(_mm512_cvtepi32_ps(_mm512_sub_epi32(c1_i32_0, comp1_0)));
    fVec y1 = fVec(_mm512_cvtepi32_ps(_mm512_sub_epi32(c1_i32_1, comp1_1)));
    
    // Load weight scales
    fVec bs0_0 = fVec::loadu(Bs0 + d);
    fVec bs0_1 = fVec::loadu(Bs0 + d + 16);
    fVec bs1_0 = fVec::loadu(Bs1 + d);
    fVec bs1_1 = fVec::loadu(Bs1 + d + 16);
    
    // Apply scales: result = As * (C - Bcomp) * Bs
    x0 = vas * x0 * bs0_0;
    x1 = vas * x1 * bs0_1;
    y0 = vas * y0 * bs1_0;
    y1 = vas * y1 * bs1_1;
    
    // silu: x / (1 + exp(-x))
    x0 = x0 / (one + x0.neg().exp_u20());
    x1 = x1 / (one + x1.neg().exp_u20());
    
    // mul and store
    x0 = x0 * y0;
    x1 = x1 * y1;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  
  // Handle remainder
  for (; d < N; ++d) {
    float x = a_scale * static_cast<float>(C0[d] - Bcomp0[d]) * Bs0[d];
    float y = a_scale * static_cast<float>(C1[d] - Bcomp1[d]) * Bs1[d];
    float silu_x = x / (1.f + std::exp(-x));
    out[d] = static_cast<scalar_t>(silu_x * y);
  }
#else
  // Scalar fallback
  for (int64_t d = 0; d < N; ++d) {
    float x = a_scale * static_cast<float>(C0[d] - Bcomp0[d]) * Bs0[d];
    float y = a_scale * static_cast<float>(C1[d] - Bcomp1[d]) * Bs1[d];
    float silu_x = x / (1.f + std::exp(-x));
    out[d] = static_cast<scalar_t>(silu_x * y);
  }
#endif
}

// apply scales and multiply by topk_weight in one pass
// Fuses: C_scaled = As * (C - Bcomp) * Bs * weight
template <typename scalar_t>
inline void scale_and_weight_fused(
    scalar_t* __restrict__ out,
    const int32_t* __restrict__ C,
    float a_scale,
    const float* __restrict__ Bs,
    const int32_t* __restrict__ Bcomp,
    float weight,
    int64_t size) {
#if defined(CPU_CAPABILITY_AVX512)
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec combined_scale = fVec(a_scale * weight);
  
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    __m512i c_i32_0 = _mm512_loadu_si512(C + d);
    __m512i c_i32_1 = _mm512_loadu_si512(C + d + 16);
    __m512i comp_0 = _mm512_loadu_si512(Bcomp + d);
    __m512i comp_1 = _mm512_loadu_si512(Bcomp + d + 16);
    
    fVec x0 = fVec(_mm512_cvtepi32_ps(_mm512_sub_epi32(c_i32_0, comp_0)));
    fVec x1 = fVec(_mm512_cvtepi32_ps(_mm512_sub_epi32(c_i32_1, comp_1)));
    
    fVec bs0 = fVec::loadu(Bs + d);
    fVec bs1 = fVec::loadu(Bs + d + 16);
    
    x0 = combined_scale * x0 * bs0;
    x1 = combined_scale * x1 * bs1;
    
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  
  for (; d < size; ++d) {
    float x = a_scale * static_cast<float>(C[d] - Bcomp[d]) * Bs[d] * weight;
    out[d] = static_cast<scalar_t>(x);
  }
#else
  float combined = a_scale * weight;
  for (int64_t d = 0; d < size; ++d) {
    float x = combined * static_cast<float>(C[d] - Bcomp[d]) * Bs[d];
    out[d] = static_cast<scalar_t>(x);
  }
#endif
}

// ============================================================================
// Static cache for reordered weights - persists across function calls
// Key: (packed_weights_ptr, E, N, K) -> reordered oneDNN memory objects
// ============================================================================
struct WeightCacheKey {
  const void* ptr;
  int64_t E, N, K;
  bool is_w1;  // true for w1, false for w2
  
  bool operator==(const WeightCacheKey& other) const {
    return ptr == other.ptr && E == other.E && N == other.N && K == other.K && is_w1 == other.is_w1;
  }
};

struct WeightCacheKeyHash {
  size_t operator()(const WeightCacheKey& k) const {
    size_t h = std::hash<const void*>{}(k.ptr);
    h ^= std::hash<int64_t>{}(k.E) << 1;
    h ^= std::hash<int64_t>{}(k.N) << 2;
    h ^= std::hash<int64_t>{}(k.K) << 3;
    h ^= std::hash<bool>{}(k.is_w1) << 4;
    return h;
  }
};

struct CachedWeights {
  std::vector<dnnl::memory> expert_weights;  // indexed by expert_id
  std::vector<std::vector<int32_t>> expert_comp;  // compensation values
  dnnl::memory::desc opt_md;  // optimal memory descriptor
  bool initialized = false;
};

// Thread-safe static cache
static std::mutex g_weight_cache_mutex;
static std::unordered_map<WeightCacheKey, CachedWeights, WeightCacheKeyHash> g_weight_cache;

#endif

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

template <>
inline void copy_stub<uint8_t>(uint8_t* __restrict__ out, const uint8_t* __restrict__ input, int64_t size) {
  // size might be 64x + 32
  std::memcpy(out, input, size * sizeof(uint8_t));
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * weight_vec;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

// acc from [topk, K] to [K]
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  if (topk == 1) {
    // do copy for topk = 1
    copy_stub(out, input, K);
  } else {
    // do sum for topk != 1
    int64_t d;
#pragma GCC unroll 4
    for (d = 0; d <= K - kVecSize; d += kVecSize) {
      fVec sum_fvec0 = fVec(0.f);
      fVec sum_fvec1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x_bvec = bVec::loadu(input + t * K + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec0 += x_fvec0;
        sum_fvec1 += x_fvec1;
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

// out = input + input2 * scale
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
    // update As
    vas = _mm512_set1_ps(As[m]);
    // C = As * (C - Bcomp) * Bs
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
    // update As
    vas = _mm512_set1_ps(As[m]);
    // C = As * (C - Bcomp) * Bs
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

/// gemm for w13
template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B0,
      const int8_t* __restrict__ B1,
      scalar_t* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs0,
      const float* __restrict__ Bs1,
      const int32_t* __restrict__ Bcomp0,
      const int32_t* __restrict__ Bcomp1,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B0,
      const int8_t* __restrict__ B1,
      at::BFloat16* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs0,
      const float* __restrict__ Bs1,
      const int32_t* __restrict__ Bcomp0,
      const int32_t* __restrict__ Bcomp1,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    __m512i va;
    __m512i vb0[COLS];
    __m512i vb1[COLS];
    __m512i vc0[ROWS * COLS];
    __m512i vc1[ROWS * COLS];
    __m512i vcomp0[COLS];
    __m512i vcomp1[COLS];
    __m512 vas;
    __m512 vbs0[COLS];
    __m512 vbs1[COLS];

    auto loadc = [&](auto i) {
      vc0[i] = _mm512_set1_epi32(0);
      vc1[i] = _mm512_set1_epi32(0);
    };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K4 = K >> 2;
    const int64_t lda4 = lda >> 2;
    const int64_t ldb4 = ldb;  // ldb * 4 >> 2;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A);
    const int32_t* b0_ptr = reinterpret_cast<const int32_t*>(B0);
    const int32_t* b1_ptr = reinterpret_cast<const int32_t*>(B1);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
      }
      if constexpr (row == 0) {
        vb0[col] = _mm512_loadu_si512(b0_ptr + k * ldb4 + col * 16);
        vb1[col] = _mm512_loadu_si512(b1_ptr + k * ldb4 + col * 16);
      }
      vc0[i] = _mm512_dpbusd_epi32(vc0[i], va, vb0[col]);
      vc1[i] = _mm512_dpbusd_epi32(vc1[i], va, vb1[col]);
    };
    for (int64_t k = 0; k < K4; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto scalec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // load a scale
      if constexpr (col == 0) {
        vas = _mm512_set1_ps(As[row]);
      }
      // load b scale and vcomp
      if constexpr (row == 0) {
        vbs0[col] = _mm512_loadu_ps(Bs0 + col * 16);
        vbs1[col] = _mm512_loadu_ps(Bs1 + col * 16);
        vcomp0[col] = _mm512_loadu_si512(Bcomp0 + col * 16);
        vcomp1[col] = _mm512_loadu_si512(Bcomp1 + col * 16);
      }
      __m512 c0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc0[i], vcomp0[col]));
      __m512 c1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc1[i], vcomp1[col]));
      vc0[i] = _mm512_castps_si512(_mm512_mul_ps(_mm512_mul_ps(c0, vas), vbs0[col]));
      vc1[i] = _mm512_castps_si512(_mm512_mul_ps(_mm512_mul_ps(c1, vas), vbs1[col]));
    };
    Unroll<ROWS * COLS>{}(scalec);

    using Vec = at::vec::Vectorized<float>;
    const Vec one = Vec(1.f);
    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      // for COLS = 2, 4 use 512bit store
      if constexpr (col % 2 == 0) {
        Vec x0 = _mm512_castsi512_ps(vc0[row * COLS + col + 0]);
        Vec x1 = _mm512_castsi512_ps(vc0[row * COLS + col + 1]);
        Vec y0 = _mm512_castsi512_ps(vc1[row * COLS + col + 0]);
        Vec y1 = _mm512_castsi512_ps(vc1[row * COLS + col + 1]);
        // silu
        x0 = x0 / (one + x0.neg().exp_u20());
        x1 = x1 / (one + x1.neg().exp_u20());
        // mul
        x0 = x0 * y0;
        x1 = x1 * y1;

        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>((C + row * ldc + col * 16)),
            (__m512i)(_mm512_cvtne2ps_pbh(__m512(x1), __m512(x0))));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_VNNI(MB_SIZE, NB_SIZE)      \
  tinygemm_kernel_vnni<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                  \
      B0 + nb_start * 4,                                   \
      B1 + nb_start * 4,                                   \
      C + mb_start * ldc + nb_start,                       \
      As + mb_start,                                       \
      Bs0 + nb_start,                                      \
      Bs1 + nb_start,                                      \
      Bcomp0 + nb_start,                                   \
      Bcomp1 + nb_start,                                   \
      K,                                                   \
      lda,                                                 \
      ldb,                                                 \
      ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B0,
    const int8_t* __restrict__ B1,
    scalar_t* __restrict__ C,
    const float* __restrict__ As,
    const float* __restrict__ Bs0,
    const float* __restrict__ Bs1,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  const int32_t* Bcomp0 = reinterpret_cast<const int32_t*>(B0 + block_size_n() * K);
  const int32_t* Bcomp1 = reinterpret_cast<const int32_t*>(B1 + block_size_n() * K);

  // pattern: 1-(2+2)-(8+8)
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 32;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_VNNI(1, 32);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_VNNI(2, 32);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_VNNI(3, 32);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_VNNI(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

/// gemm for w2
template <typename scalar_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni2 {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B,
      float* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs,
      const int32_t* __restrict__ Bcomp,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_vnni2<at::BFloat16, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const uint8_t* __restrict__ A,
      const int8_t* __restrict__ B,
      float* __restrict__ C,
      const float* __restrict__ As,
      const float* __restrict__ Bs,
      const int32_t* __restrict__ Bcomp,
      int64_t K,
      int64_t lda,
      int64_t ldb,
      int64_t ldc) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;
    static_assert(COLS % 2 == 0);

    __m512i va;
    __m512i vb[COLS];
    __m512i vc[ROWS * COLS];
    __m512i vcomp[COLS];
    __m512 vas;
    __m512 vbs[COLS];

    auto loadc = [&](auto i) { vc[i] = _mm512_set1_epi32(0); };
    Unroll<ROWS * COLS>{}(loadc);

    const int64_t K4 = K >> 2;
    const int64_t lda4 = lda >> 2;
    const int64_t ldb4 = ldb;  // ldb * 4 >> 2;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A);
    const int32_t* b_ptr = reinterpret_cast<const int32_t*>(B);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_epi32(a_ptr[row * lda4 + k]);
      }
      if constexpr (row == 0) {
        vb[col] = _mm512_loadu_si512(b_ptr + k * ldb4 + col * 16);
      }
      vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
    };
    for (int64_t k = 0; k < K4; ++k) {
      Unroll<ROWS * COLS>{}(compute, k);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // load a scale
      if constexpr (col == 0) {
        vas = _mm512_set1_ps(As[row]);
      }
      // load b scale and vcomp per 2 vectors
      // also load bias if any
      if constexpr (row == 0) {
        if constexpr (col % 2 == 0) {
          vbs[col + 0] = _mm512_loadu_ps(Bs + col * 16);
          vbs[col + 1] = _mm512_loadu_ps(Bs + col * 16 + 16);
          vcomp[col + 0] = _mm512_loadu_si512(Bcomp + col * 16);
          vcomp[col + 1] = _mm512_loadu_si512(Bcomp + col * 16 + 16);
        }
      }
      __m512 x = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc[i], vcomp[col]));
      x = _mm512_mul_ps(_mm512_mul_ps(x, vas), vbs[col]);
      _mm512_storeu_ps(reinterpret_cast<__m512*>(C + row * ldc + col * 16), x);
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_VNNI2(MB_SIZE, NB_SIZE)      \
  tinygemm_kernel_vnni2<scalar_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                   \
      B + nb_start * 4,                                     \
      C + mb_start * ldc + nb_start,                        \
      As + mb_start,                                        \
      Bs + nb_start,                                        \
      Bcomp + nb_start,                                     \
      K,                                                    \
      lda,                                                  \
      ldb,                                                  \
      ldc);

template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc) {
  // B compensation
  const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + block_size_n() * K);

  // pattern: 1-4-16
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size >> 4) {
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_VNNI2(1, 32);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_VNNI2(2, 32);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_VNNI2(3, 32);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_VNNI2(4, 32);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

}  // anonymous namespace

#if !defined(SGLANG_USE_ONEDNN_MOE) || SGLANG_USE_ONEDNN_MOE == 0
// ============================================================================
// Original implementation using brgemm/tinygemm
// ============================================================================
template <typename scalar_t>
void fused_experts_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
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
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // strides for w1: [E, 2N, K]
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  // K and N are packed for int8
  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);

  const int64_t stride_e = 2 * N * packed_K;
  const int64_t stride_n = packed_K;

  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  const bool use_brgemm = can_use_brgemm<int8_t>(avg_M);

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    uint8_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    int32_t* __restrict__ C0 = reinterpret_cast<int32_t*>(C_tmp) + tid * 2 * BLOCK_M * BLOCK_N;
    int32_t* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;

    alignas(64) float As[BLOCK_M];

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      // nb_upper from top half and nb_lower from bottom half
      int64_t nb_upper = nb, nb_lower = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);

      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const int8_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb_upper * BLOCK_N * stride_n;
      const int8_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + nb_lower * BLOCK_N * stride_n;
      const float* __restrict__ Bs0 = w1s + expert_id * 2 * N + nb_upper * BLOCK_N;
      const float* __restrict__ Bs1 = w1s + expert_id * 2 * N + nb_lower * BLOCK_N;

      // 1.a load A
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub(A + m * K, Aq_tmp + index * K, K);
        As[m] = As_tmp[index];
      }

      if (use_brgemm) {
        // 1.b gemm: C0 = A @ B0
        at::native::cpublas::brgemm(
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
        at::native::cpublas::brgemm(
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
        const int64_t offset = offsets[mb];
        silu_and_mul<scalar_t, BLOCK_N>(
            ic1 + offset * N + nb * BLOCK_N, C0, C1, As, Bs0, Bs1, Bcomp0, Bcomp1, m_size, N);
      } else {
        // fused 1.bcd: silu_and_mul(A @ B0, A @ B1)
        const int64_t offset = offsets[mb];
        tinygemm_kernel(
            /* A     */ A,
            /* B0    */ B0,
            /* B1    */ B1,
            /* C     */ ic1 + offset * N + nb * BLOCK_N,
            /* As    */ As,
            /* Bs0   */ Bs0,
            /* Bs1   */ Bs1,
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ N);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * N, As_tmp[m], ic1 + m * N, N);
    }
  });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * packed_N;
  const int64_t stride_oc = packed_N;

  // parallel on [MB2, NB2]
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // get local pointers
    int tid = get_thread_num();
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    int32_t* __restrict__ C32 = reinterpret_cast<int32_t*>(C + BLOCK_M * BLOCK_N);

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const uint8_t* __restrict__ A = Aq_tmp + offsets[mb] * N;
      const float* __restrict__ As = As_tmp + offsets[mb];
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const int8_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const float* __restrict__ Bs = w2s + expert_id * K + nb * BLOCK_N;

      // 2.a gemm: C = A @ B
      if (use_brgemm) {
        at::native::cpublas::brgemm(
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
      } else {
        tinygemm_kernel<scalar_t>(
            /* A     */ A,
            /* B     */ B,
            /* C     */ C,
            /* As    */ As,
            /* Bs    */ Bs,
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N);
      }

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 3: out = intermediate_cache2.sum(dim=1)
  //   from [M, topk, K] to [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#elif SGLANG_USE_ONEDNN_MOE == 1
// ============================================================================
// oneDNN-based implementation (sequential expert loop)
// Uses oneDNN matmul primitive with format_tag::BA16a64b4a for weights optimization
// Key optimizations:
// 1. STATIC weight cache - reorder weights once, reuse across all calls
// 2. Cache reordered weights in oneDNN memory objects for reuse
// 3. Fused scale application with silu_and_mul using SIMD
// 4. Vectorized compensation and scale application
// 5. Pre-allocated buffers outside the expert loop
// Note: oneDNN has its own internal primitive cache, so we don't need to
//       cache primitives ourselves - just create them with the same descriptors
// ============================================================================

template <typename scalar_t>
void fused_experts_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
  using namespace dnnl;

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * K, As_tmp[m], input + m * K, K);
    }
  });

  // Initialize oneDNN engine and stream
  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);

  // strides for w1: [E, 2N, K] - w1 contains both gate and up projections
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  // K and N are packed for int8 (VNNI format with compensation)
  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);

  const int64_t stride_e = 2 * N * packed_K;

  // ============================================================================
  // Get or create cached weights - STATIC CACHE for cross-call reuse
  // ============================================================================
  
  // Create cache keys for w1 and w2
  WeightCacheKey w1_key{packed_w1, E, N, K, true};
  WeightCacheKey w2_key{packed_w2, E, N, K, false};
  
  CachedWeights* w1_cache = nullptr;
  CachedWeights* w2_cache = nullptr;
  
  {
    std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
    
    // Check and initialize w1 cache
    auto& w1_cached = g_weight_cache[w1_key];
    if (!w1_cached.initialized) {
      // First time seeing this weight - need to unpack and reorder ALL experts
      w1_cached.expert_weights.resize(E);
      w1_cached.expert_comp.resize(E);
      
      // Create primitive descriptor for w1 to get optimal weight format
      auto w1_a_md = memory::desc({BLOCK_M, K}, memory::data_type::u8, memory::format_tag::ab);
      auto w1_b_md = memory::desc({K, 2 * N}, memory::data_type::s8, memory::format_tag::BA16a64b4a);
      auto w1_c_md = memory::desc({BLOCK_M, 2 * N}, memory::data_type::s32, memory::format_tag::ab);
      matmul::primitive_desc w1_pd(eng, w1_a_md, w1_b_md, w1_c_md);
      w1_cached.opt_md = w1_pd.weights_desc();
      
      // Memory descriptor for source (unpacked) weights
      memory::dims w1_b_dims = {K, 2 * N};
      memory::dims w1_b_strides = {1, K};  // Transposed layout
      auto w1_src_md = memory::desc(w1_b_dims, memory::data_type::s8, w1_b_strides);
      
      // Unpack and reorder ALL experts (not just used ones)
      std::vector<int8_t> unpacked(2 * N * K);
      for (int32_t expert_id = 0; expert_id < E; ++expert_id) {
        const int8_t* B_packed = packed_w1 + expert_id * stride_e;
        
        // Unpack weights
        unpack_vnni_int8_no_transpose<BLOCK_N>(unpacked.data(), B_packed, 2 * N, K);
        
        // Extract compensation values
        w1_cached.expert_comp[expert_id].resize(2 * N);
        const int64_t num_blocks = (2 * N + BLOCK_N - 1) / BLOCK_N;
        for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
          int64_t row_start = block_idx * BLOCK_N;
          int64_t row_end = std::min(row_start + BLOCK_N, (int64_t)(2 * N));
          const int8_t* block_ptr = B_packed + block_idx * BLOCK_N * (K + 4);
          const int32_t* comp_ptr = reinterpret_cast<const int32_t*>(block_ptr + BLOCK_N * K);
          for (int64_t i = 0; i < row_end - row_start; ++i) {
            w1_cached.expert_comp[expert_id][row_start + i] = comp_ptr[i];
          }
        }
        
        // Reorder to optimal format
        memory w1_src(w1_src_md, eng, unpacked.data());
        w1_cached.expert_weights[expert_id] = memory(w1_cached.opt_md, eng);
        reorder(w1_src, w1_cached.expert_weights[expert_id]).execute(s, w1_src, w1_cached.expert_weights[expert_id]);
      }
      s.wait();
      w1_cached.initialized = true;
    }
    w1_cache = &w1_cached;
    
    // Check and initialize w2 cache
    const int64_t OC = K;
    const int64_t IC = N;
    const int64_t stride_e2 = OC * packed_N;
    
    auto& w2_cached = g_weight_cache[w2_key];
    if (!w2_cached.initialized) {
      w2_cached.expert_weights.resize(E);
      w2_cached.expert_comp.resize(E);
      
      auto w2_a_md = memory::desc({BLOCK_M, IC}, memory::data_type::u8, memory::format_tag::ab);
      auto w2_b_md = memory::desc({IC, OC}, memory::data_type::s8, memory::format_tag::BA16a64b4a);
      auto w2_c_md = memory::desc({BLOCK_M, OC}, memory::data_type::s32, memory::format_tag::ab);
      matmul::primitive_desc w2_pd(eng, w2_a_md, w2_b_md, w2_c_md);
      w2_cached.opt_md = w2_pd.weights_desc();
      
      memory::dims w2_b_dims = {IC, OC};
      memory::dims w2_b_strides = {1, IC};
      auto w2_src_md = memory::desc(w2_b_dims, memory::data_type::s8, w2_b_strides);
      
      std::vector<int8_t> unpacked(OC * IC);
      for (int32_t expert_id = 0; expert_id < E; ++expert_id) {
        const int8_t* B_packed = packed_w2 + expert_id * stride_e2;
        
        unpack_vnni_int8_no_transpose<BLOCK_N>(unpacked.data(), B_packed, OC, IC);
        
        w2_cached.expert_comp[expert_id].resize(OC);
        const int64_t num_blocks_w2 = (OC + BLOCK_N - 1) / BLOCK_N;
        for (int64_t block_idx = 0; block_idx < num_blocks_w2; ++block_idx) {
          int64_t row_start = block_idx * BLOCK_N;
          int64_t row_end = std::min(row_start + BLOCK_N, (int64_t)OC);
          const int8_t* block_ptr = B_packed + block_idx * BLOCK_N * (IC + 4);
          const int32_t* comp_ptr = reinterpret_cast<const int32_t*>(block_ptr + BLOCK_N * IC);
          for (int64_t i = 0; i < row_end - row_start; ++i) {
            w2_cached.expert_comp[expert_id][row_start + i] = comp_ptr[i];
          }
        }
        
        memory w2_src(w2_src_md, eng, unpacked.data());
        w2_cached.expert_weights[expert_id] = memory(w2_cached.opt_md, eng);
        reorder(w2_src, w2_cached.expert_weights[expert_id]).execute(s, w2_src, w2_cached.expert_weights[expert_id]);
      }
      s.wait();
      w2_cached.initialized = true;
    }
    w2_cache = &w2_cached;
  }
  
  const int64_t OC = K;
  const int64_t IC = N;

  // ============================================================================
  // Pre-compute expert batches and find max_m_size
  // ============================================================================
  int64_t max_m_size = 0;
  std::vector<std::tuple<int64_t, int64_t, int32_t>> expert_batches;  // (mb_start, mb_end, expert_id)
  
  for (int64_t mb_start = 0; mb_start < MB; ) {
    int32_t current_expert = expert_ids[mb_start];
    int64_t mb_end = mb_start + 1;
    
    while (mb_end < MB && expert_ids[mb_end] == current_expert) {
      mb_end++;
    }
    
    int64_t token_start = offsets[mb_start];
    int64_t token_end = offsets[mb_end];
    int64_t m_size = token_end - token_start;
    
    if (m_size > 0) {
      expert_batches.emplace_back(mb_start, mb_end, current_expert);
      max_m_size = std::max(max_m_size, m_size);
    }
    
    mb_start = mb_end;
  }
  
  if (max_m_size == 0) {
    // No tokens to process, just zero output
    at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        std::memset(output + m * K, 0, K * sizeof(scalar_t));
      }
    });
    return;
  }
  
  // Pad max_m_size to BLOCK_M for better AMX utilization
  const int64_t padded_m = div_up(max_m_size, BLOCK_M) * BLOCK_M;

  // ============================================================================
  // Create matmul primitives - oneDNN internally caches primitives with same descriptors
  // ============================================================================
  auto w1_a_md = memory::desc({padded_m, K}, memory::data_type::u8, memory::format_tag::ab);
  auto w1_b_md = memory::desc({K, 2 * N}, memory::data_type::s8, memory::format_tag::BA16a64b4a);
  auto w1_c_md = memory::desc({padded_m, 2 * N}, memory::data_type::s32, memory::format_tag::ab);
  matmul::primitive_desc w1_pd(eng, w1_a_md, w1_b_md, w1_c_md);
  matmul w1_prim(w1_pd);
  
  auto w2_a_md = memory::desc({padded_m, IC}, memory::data_type::u8, memory::format_tag::ab);
  auto w2_b_md = memory::desc({IC, OC}, memory::data_type::s8, memory::format_tag::BA16a64b4a);
  auto w2_c_md = memory::desc({padded_m, OC}, memory::data_type::s32, memory::format_tag::ab);
  matmul::primitive_desc w2_pd(eng, w2_a_md, w2_b_md, w2_c_md);
  matmul w2_prim(w2_pd);

  // ============================================================================
  // Pre-allocate ALL buffers OUTSIDE the expert loop
  // This eliminates repeated memory allocation overhead
  // ============================================================================
  
  // Buffers for Stage 1 (w1 matmul)
  std::vector<uint8_t> A_padded_w1(padded_m * K, 0);
  std::vector<int32_t> C_combined(padded_m * 2 * N);
  std::vector<float> As_padded_w1(padded_m);
  std::vector<int32_t> gathered_ids_w1(padded_m);
  
  // Buffers for Stage 2 (w2 matmul) - reuse where possible
  std::vector<uint8_t> A_padded_w2(padded_m * IC, 0);
  std::vector<int32_t> C_s32(padded_m * OC);
  std::vector<float> As_padded_w2(padded_m);
  std::vector<int32_t> gathered_ids_w2(padded_m);
  
  // Create memory objects once - these wrap the pre-allocated buffers
  auto a_mem_w1 = memory(w1_a_md, eng, A_padded_w1.data());
  auto c_mem_w1 = memory(w1_c_md, eng, C_combined.data());
  auto a_mem_w2 = memory(w2_a_md, eng, A_padded_w2.data());
  auto c_mem_w2 = memory(w2_c_md, eng, C_s32.data());

  // ============================================================================
  // Stage 1: First matmul (w1) - gate and up projections with silu_and_mul
  // ============================================================================
  
  for (const auto& batch : expert_batches) {
    int64_t mb_start = std::get<0>(batch);
    int64_t mb_end = std::get<1>(batch);
    int32_t current_expert = std::get<2>(batch);
    
    int64_t token_start = offsets[mb_start];
    int64_t token_end = offsets[mb_end];
    int64_t m_size = token_end - token_start;
    
    // Check for contiguous access pattern - common case with topk=1 and single expert
    // If tokens are in order, we can avoid the expensive gather
    bool is_contiguous = true;
    int32_t first_index = -1;
    int64_t check_count = 0;
    
    for (int64_t mb = mb_start; mb < mb_end && is_contiguous; ++mb) {
      const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
      int64_t mb_size = offsets[mb + 1] - offsets[mb];
      for (int64_t m = 0; m < mb_size && is_contiguous; ++m) {
        int32_t sorted_id = A_ids_mb[m];
        int32_t index = sorted_id / topk;
        if (first_index < 0) {
          first_index = index;
        }
        // Check if index == first_index + check_count (contiguous)
        if (index != first_index + check_count) {
          is_contiguous = false;
        }
        check_count++;
      }
    }
    
    // Pointer to actual input data (either original or gathered)
    const uint8_t* A_src_w1 = nullptr;
    const float* As_src_w1 = nullptr;
    
    if (is_contiguous && first_index >= 0) {
      // Contiguous access - potentially avoid gather entirely
      
      // Still need to fill gathered_ids for scatter
      int64_t gathered = 0;
      for (int64_t mb = mb_start; mb < mb_end; ++mb) {
        const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
        int64_t mb_size = offsets[mb + 1] - offsets[mb];
        for (int64_t m = 0; m < mb_size; ++m) {
          gathered_ids_w1[gathered] = A_ids_mb[m];
          gathered++;
        }
      }
      
      if (m_size == padded_m) {
        // Perfect alignment - use original buffer directly (zero copy!)
        A_src_w1 = Aq_tmp + first_index * K;
        As_src_w1 = As_tmp + first_index;
      } else {
        // Need padding - must copy to padded buffer
        std::memcpy(A_padded_w1.data(), Aq_tmp + first_index * K, m_size * K);
        std::memset(A_padded_w1.data() + m_size * K, 0, (padded_m - m_size) * K);
        std::memcpy(As_padded_w1.data(), As_tmp + first_index, m_size * sizeof(float));
        A_src_w1 = A_padded_w1.data();
        As_src_w1 = As_padded_w1.data();
      }
    } else {
      // Non-contiguous access - use parallel gather for large sizes
      A_src_w1 = A_padded_w1.data();
      As_src_w1 = As_padded_w1.data();
      
      // Build index mapping first (small loop)
      std::vector<std::pair<int32_t, int32_t>> index_map(m_size);  // (src_index, sorted_id)
      int64_t gathered = 0;
      for (int64_t mb = mb_start; mb < mb_end; ++mb) {
        const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
        int64_t mb_size = offsets[mb + 1] - offsets[mb];
        for (int64_t m = 0; m < mb_size; ++m) {
          int32_t sorted_id = A_ids_mb[m];
          int32_t index = sorted_id / topk;
          index_map[gathered] = {index, sorted_id};
          gathered_ids_w1[gathered] = sorted_id;
          gathered++;
        }
      }
      
      // Parallel gather for large token counts
      constexpr int64_t PARALLEL_THRESHOLD = 64;
      if (m_size >= PARALLEL_THRESHOLD) {
        at::parallel_for(0, m_size, 0, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            int32_t index = index_map[i].first;
            std::memcpy(A_padded_w1.data() + i * K, Aq_tmp + index * K, K);
            As_padded_w1[i] = As_tmp[index];
          }
        });
      } else {
        // Sequential gather for small sizes (better cache behavior)
        for (int64_t i = 0; i < m_size; ++i) {
          int32_t index = index_map[i].first;
          std::memcpy(A_padded_w1.data() + i * K, Aq_tmp + index * K, K);
          As_padded_w1[i] = As_tmp[index];
        }
      }
      
      // Zero-pad remaining rows if needed
      if (m_size < padded_m) {
        std::memset(A_padded_w1.data() + m_size * K, 0, (padded_m - m_size) * K);
      }
    }

    // Get cached weights
    auto& b_mem = w1_cache->expert_weights[current_expert];

    // Update memory data handle if source changed (for contiguous optimization)
    a_mem_w1.set_data_handle(const_cast<uint8_t*>(A_src_w1));

    // Execute matmul - oneDNN caches the primitive internally
    w1_prim.execute(s, {
        {DNNL_ARG_SRC, a_mem_w1},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem_w1}
    });
    s.wait();

    // Apply scales and silu_and_mul - parallelized
    const int32_t* Bcomp_w1 = w1_cache->expert_comp[current_expert].data();
    const float* Bs = w1s + current_expert * 2 * N;
    const float* Bs0 = Bs;
    const float* Bs1 = Bs + N;
    const int32_t* Bcomp0 = Bcomp_w1;
    const int32_t* Bcomp1 = Bcomp_w1 + N;
    
    at::parallel_for(0, m_size, 0, [&](int64_t m_begin, int64_t m_end_local) {
      for (int64_t m = m_begin; m < m_end_local; ++m) {
        float a_scale = As_src_w1[m];
        const int32_t* C0 = C_combined.data() + m * 2 * N;
        const int32_t* C1 = C_combined.data() + m * 2 * N + N;
        int32_t out_idx = gathered_ids_w1[m];
        
        scale_and_silu_mul_fused<scalar_t>(
            ic1 + out_idx * N,
            C0, C1,
            a_scale,
            Bs0, Bs1,
            Bcomp0, Bcomp1,
            N);
      }
    });
  }

  // stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * N, As_tmp[m], ic1 + m * N, N);
    }
  });

  // ============================================================================
  // Stage 2: Second matmul (w2) - down projection
  // ============================================================================
  
  for (const auto& batch : expert_batches) {
    int64_t mb_start = std::get<0>(batch);
    int64_t mb_end = std::get<1>(batch);
    int32_t current_expert = std::get<2>(batch);
    
    int64_t token_start = offsets[mb_start];
    int64_t token_end = offsets[mb_end];
    int64_t m_size = token_end - token_start;
    
    // Check for contiguous access pattern (common with topk=1, single expert)
    bool is_contiguous_w2 = true;
    int32_t first_sorted_id = -1;
    int64_t check_count = 0;
    
    for (int64_t mb = mb_start; mb < mb_end && is_contiguous_w2; ++mb) {
      const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
      int64_t mb_size = offsets[mb + 1] - offsets[mb];
      for (int64_t m = 0; m < mb_size && is_contiguous_w2; ++m) {
        int32_t sorted_id = A_ids_mb[m];
        if (first_sorted_id < 0) {
          first_sorted_id = sorted_id;
        }
        if (sorted_id != first_sorted_id + check_count) {
          is_contiguous_w2 = false;
        }
        check_count++;
      }
    }
    
    const uint8_t* A_src_w2 = nullptr;
    const float* As_src_w2 = nullptr;
    
    if (is_contiguous_w2 && first_sorted_id >= 0) {
      // Fill gathered_ids for scatter
      int64_t gathered = 0;
      for (int64_t mb = mb_start; mb < mb_end; ++mb) {
        const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
        int64_t mb_size = offsets[mb + 1] - offsets[mb];
        for (int64_t m = 0; m < mb_size; ++m) {
          gathered_ids_w2[gathered] = A_ids_mb[m];
          gathered++;
        }
      }
      
      if (m_size == padded_m) {
        // Perfect alignment - zero copy
        A_src_w2 = Aq_tmp + first_sorted_id * N;
        As_src_w2 = As_tmp + first_sorted_id;
      } else {
        // Need padding
        std::memcpy(A_padded_w2.data(), Aq_tmp + first_sorted_id * N, m_size * IC);
        std::memset(A_padded_w2.data() + m_size * IC, 0, (padded_m - m_size) * IC);
        std::memcpy(As_padded_w2.data(), As_tmp + first_sorted_id, m_size * sizeof(float));
        A_src_w2 = A_padded_w2.data();
        As_src_w2 = As_padded_w2.data();
      }
    } else {
      // Non-contiguous - gather
      A_src_w2 = A_padded_w2.data();
      As_src_w2 = As_padded_w2.data();
      
      std::vector<std::pair<int32_t, int32_t>> index_map_w2(m_size);
      int64_t gathered = 0;
      for (int64_t mb = mb_start; mb < mb_end; ++mb) {
        const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
        int64_t mb_size = offsets[mb + 1] - offsets[mb];
        for (int64_t m = 0; m < mb_size; ++m) {
          int32_t sorted_id = A_ids_mb[m];
          index_map_w2[gathered] = {sorted_id, sorted_id};
          gathered_ids_w2[gathered] = sorted_id;
          gathered++;
        }
      }
      
      constexpr int64_t PARALLEL_THRESHOLD = 64;
      if (m_size >= PARALLEL_THRESHOLD) {
        at::parallel_for(0, m_size, 0, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            int32_t sorted_id = index_map_w2[i].first;
            std::memcpy(A_padded_w2.data() + i * IC, Aq_tmp + sorted_id * N, IC);
            As_padded_w2[i] = As_tmp[sorted_id];
          }
        });
      } else {
        for (int64_t i = 0; i < m_size; ++i) {
          int32_t sorted_id = index_map_w2[i].first;
          std::memcpy(A_padded_w2.data() + i * IC, Aq_tmp + sorted_id * N, IC);
          As_padded_w2[i] = As_tmp[sorted_id];
        }
      }
      
      if (m_size < padded_m) {
        std::memset(A_padded_w2.data() + m_size * IC, 0, (padded_m - m_size) * IC);
      }
    }

    auto& b_mem = w2_cache->expert_weights[current_expert];

    a_mem_w2.set_data_handle(const_cast<uint8_t*>(A_src_w2));

    w2_prim.execute(s, {
        {DNNL_ARG_SRC, a_mem_w2},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem_w2}
    });
    s.wait();

    const int32_t* Bcomp_w2 = w2_cache->expert_comp[current_expert].data();
    const float* Bs = w2s + current_expert * K;

    at::parallel_for(0, m_size, 0, [&](int64_t m_begin, int64_t m_end_local) {
      for (int64_t m = m_begin; m < m_end_local; ++m) {
        float a_scale = As_src_w2[m];
        int32_t index = gathered_ids_w2[m];
        float weight = topk_weights[index];
        const int32_t* C_row = C_s32.data() + m * OC;
        
        scale_and_weight_fused<scalar_t>(
            ic2 + index * K,
            C_row,
            a_scale,
            Bs,
            Bcomp_w2,
            weight,
            OC);
      }
    });
  }

  // stage 3: out = intermediate_cache2.sum(dim=1)
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#elif SGLANG_USE_ONEDNN_MOE == 2
// ============================================================================
// oneDNN-based implementation (batched matmul - all experts in one call)
// Uses oneDNN batched matmul with format_tag::BA16a32b4a for weights optimization
// Key optimizations:
// 1. STATIC weight cache - reorder weights once, reuse across all calls
// 2. Fused scale application with silu_and_mul using SIMD
// 3. Vectorized compensation and scale application
// 4. Better parallelization over tokens instead of experts
// Note: Uses shared cache structures defined before the case 0/1/2 split
// ============================================================================

template <typename scalar_t>
void fused_experts_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
  using namespace dnnl;

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * K, As_tmp[m], input + m * K, K);
    }
  });

  // Initialize oneDNN engine and stream
  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);

  // strides for w1: [E, 2N, K] - w1 contains both gate and up projections
  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  // K and N are packed for int8 (VNNI format with compensation)
  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);

  const int64_t stride_e = 2 * N * packed_K;

  // ============================================================================
  // Get or create cached weights - STATIC CACHE for cross-call reuse
  // ============================================================================
  
  // Create cache keys for w1 and w2
  WeightCacheKey w1_key{packed_w1, E, N, K, true};
  WeightCacheKey w2_key{packed_w2, E, N, K, false};
  
  CachedWeights* w1_cache = nullptr;
  CachedWeights* w2_cache = nullptr;
  
  {
    std::lock_guard<std::mutex> lock(g_weight_cache_mutex);
    
    // Check and initialize w1 cache
    // For case 2, we store weights in 3D batched format [E, K, 2N] for batched matmul
    auto& w1_cached = g_weight_cache[w1_key];
    if (!w1_cached.initialized) {
      // First time seeing this weight - need to unpack and reorder ALL experts
      w1_cached.expert_weights.resize(1);  // Single 3D tensor for all experts
      w1_cached.expert_comp.resize(E);
      
      // Create primitive descriptor for batched w1 to get optimal weight format
      // Use format_tag::any to let oneDNN choose optimal 3D format
      auto w1_a_md = memory::desc({E, BLOCK_M, K}, memory::data_type::u8, memory::format_tag::abc);
      auto w1_b_md = memory::desc({E, K, 2 * N}, memory::data_type::s8, memory::format_tag::any);
      auto w1_c_md = memory::desc({E, BLOCK_M, 2 * N}, memory::data_type::s32, memory::format_tag::abc);
      matmul::primitive_desc w1_pd(eng, w1_a_md, w1_b_md, w1_c_md);
      w1_cached.opt_md = w1_pd.weights_desc();
      
      // Prepare source buffer in abc format [E, K, 2N]
      std::vector<int8_t> B_src(E * K * 2 * N);
      std::vector<int8_t> unpacked(2 * N * K);
      
      for (int32_t expert_id = 0; expert_id < E; ++expert_id) {
        const int8_t* B_packed = packed_w1 + expert_id * stride_e;
        
        // Unpack weights
        unpack_vnni_int8_no_transpose<BLOCK_N>(unpacked.data(), B_packed, 2 * N, K);
        
        // Extract compensation values
        w1_cached.expert_comp[expert_id].resize(2 * N);
        const int64_t num_blocks = (2 * N + BLOCK_N - 1) / BLOCK_N;
        for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
          int64_t row_start = block_idx * BLOCK_N;
          int64_t row_end = std::min(row_start + BLOCK_N, (int64_t)(2 * N));
          const int8_t* block_ptr = B_packed + block_idx * BLOCK_N * (K + 4);
          const int32_t* comp_ptr = reinterpret_cast<const int32_t*>(block_ptr + BLOCK_N * K);
          for (int64_t i = 0; i < row_end - row_start; ++i) {
            w1_cached.expert_comp[expert_id][row_start + i] = comp_ptr[i];
          }
        }
        
        // Transpose from [2N, K] to [K, 2N] and store in 3D buffer
        int8_t* B_expert = B_src.data() + expert_id * K * 2 * N;
        for (int64_t k = 0; k < K; ++k) {
          for (int64_t n = 0; n < 2 * N; ++n) {
            B_expert[k * 2 * N + n] = unpacked[n * K + k];
          }
        }
      }
      
      // Reorder entire 3D tensor to optimal format
      auto w1_src_md = memory::desc({E, K, 2 * N}, memory::data_type::s8, memory::format_tag::abc);
      memory w1_src(w1_src_md, eng, B_src.data());
      w1_cached.expert_weights[0] = memory(w1_cached.opt_md, eng);
      reorder(w1_src, w1_cached.expert_weights[0]).execute(s, w1_src, w1_cached.expert_weights[0]);
      s.wait();
      w1_cached.initialized = true;
    }
    w1_cache = &w1_cached;
    
    // Check and initialize w2 cache
    // For case 2, we store weights in 3D batched format [E, N, K] for batched matmul
    const int64_t OC = K;
    const int64_t IC = N;
    const int64_t stride_e2 = OC * packed_N;
    
    auto& w2_cached = g_weight_cache[w2_key];
    if (!w2_cached.initialized) {
      w2_cached.expert_weights.resize(1);  // Single 3D tensor for all experts
      w2_cached.expert_comp.resize(E);
      
      // Create primitive descriptor for batched w2 to get optimal weight format
      auto w2_a_md = memory::desc({E, BLOCK_M, IC}, memory::data_type::u8, memory::format_tag::abc);
      auto w2_b_md = memory::desc({E, IC, OC}, memory::data_type::s8, memory::format_tag::any);
      auto w2_c_md = memory::desc({E, BLOCK_M, OC}, memory::data_type::s32, memory::format_tag::abc);
      matmul::primitive_desc w2_pd(eng, w2_a_md, w2_b_md, w2_c_md);
      w2_cached.opt_md = w2_pd.weights_desc();
      
      // Prepare source buffer in abc format [E, IC, OC]
      std::vector<int8_t> B_src(E * IC * OC);
      std::vector<int8_t> unpacked(OC * IC);
      
      for (int32_t expert_id = 0; expert_id < E; ++expert_id) {
        const int8_t* B_packed = packed_w2 + expert_id * stride_e2;
        
        unpack_vnni_int8_no_transpose<BLOCK_N>(unpacked.data(), B_packed, OC, IC);
        
        w2_cached.expert_comp[expert_id].resize(OC);
        const int64_t num_blocks_w2 = (OC + BLOCK_N - 1) / BLOCK_N;
        for (int64_t block_idx = 0; block_idx < num_blocks_w2; ++block_idx) {
          int64_t row_start = block_idx * BLOCK_N;
          int64_t row_end = std::min(row_start + BLOCK_N, (int64_t)OC);
          const int8_t* block_ptr = B_packed + block_idx * BLOCK_N * (IC + 4);
          const int32_t* comp_ptr = reinterpret_cast<const int32_t*>(block_ptr + BLOCK_N * IC);
          for (int64_t i = 0; i < row_end - row_start; ++i) {
            w2_cached.expert_comp[expert_id][row_start + i] = comp_ptr[i];
          }
        }
        
        // Transpose from [OC, IC] to [IC, OC] and store in 3D buffer
        int8_t* B_expert = B_src.data() + expert_id * IC * OC;
        for (int64_t ic = 0; ic < IC; ++ic) {
          for (int64_t oc = 0; oc < OC; ++oc) {
            B_expert[ic * OC + oc] = unpacked[oc * IC + ic];
          }
        }
      }
      
      // Reorder entire 3D tensor to optimal format
      auto w2_src_md = memory::desc({E, IC, OC}, memory::data_type::s8, memory::format_tag::abc);
      memory w2_src(w2_src_md, eng, B_src.data());
      w2_cached.expert_weights[0] = memory(w2_cached.opt_md, eng);
      reorder(w2_src, w2_cached.expert_weights[0]).execute(s, w2_src, w2_cached.expert_weights[0]);
      s.wait();
      w2_cached.initialized = true;
    }
    w2_cache = &w2_cached;
  }

  const int64_t OC = K;
  const int64_t IC = N;

  // ============================================================================
  // Stage 1: First matmul (w1) - gate and up projections with silu_and_mul
  // TRUE BATCHED MATMUL: All E experts processed in one batched GEMM call
  // Uses 3D tensors: A[E, padded_m, K] x B[E, K, 2N] = C[E, padded_m, 2N]
  // ============================================================================
  
  // Find the maximum M size across all expert batches for uniform padding
  int64_t max_m_size = 0;
  std::vector<std::tuple<int64_t, int64_t, int32_t>> expert_batches;  // (mb_start, mb_end, expert_id)
  std::vector<int64_t> expert_m_sizes(E, 0);  // Track m_size per expert
  
  for (int64_t mb_start = 0; mb_start < MB; ) {
    int32_t current_expert = expert_ids[mb_start];
    int64_t mb_end = mb_start + 1;
    
    while (mb_end < MB && expert_ids[mb_end] == current_expert) {
      mb_end++;
    }
    
    int64_t token_start = offsets[mb_start];
    int64_t token_end = offsets[mb_end];
    int64_t m_size = token_end - token_start;
    
    if (m_size > 0) {
      expert_batches.emplace_back(mb_start, mb_end, current_expert);
      expert_m_sizes[current_expert] = m_size;
      max_m_size = std::max(max_m_size, m_size);
    }
    
    mb_start = mb_end;
  }
  
  const int64_t num_active_experts = expert_batches.size();
  
  if (max_m_size > 0 && num_active_experts > 0) {
    // Pad max_m_size to BLOCK_M for better AMX utilization
    int64_t padded_m = div_up(max_m_size, BLOCK_M) * BLOCK_M;
    
    // Create 3D batched matmul primitive: [E, padded_m, K] x [E, K, 2N] = [E, padded_m, 2N]
    // Use ALL E experts to match the cached weight tensor shape
    // Inactive experts will have zero activations and produce zero outputs
    auto a_md = memory::desc({E, padded_m, K}, memory::data_type::u8, memory::format_tag::abc);
    auto c_md = memory::desc({E, padded_m, 2 * N}, memory::data_type::s32, memory::format_tag::abc);

    // Use the cached optimal weight memory descriptor
    matmul::primitive_desc pd(eng, a_md, w1_cache->opt_md, c_md);
    matmul prim(pd);
    
    // Allocate batched buffers for ALL E experts
    std::vector<uint8_t> A_batched(E * padded_m * K, 0);  // Zero-initialized for inactive experts
    std::vector<int32_t> C_batched(E * padded_m * 2 * N, 0);
    std::vector<float> As_batched(E * padded_m, 0.f);
    std::vector<std::vector<int32_t>> gathered_ids_per_expert(E);
    
    // Gather inputs for active experts only (inactive experts stay zero)
    for (const auto& batch : expert_batches) {
      int64_t mb_start = std::get<0>(batch);
      int64_t mb_end = std::get<1>(batch);
      int32_t current_expert = std::get<2>(batch);
      
      int64_t m_size = expert_m_sizes[current_expert];
      gathered_ids_per_expert[current_expert].resize(m_size);
      
      // Gather input activations for this expert
      uint8_t* A_expert = A_batched.data() + current_expert * padded_m * K;
      float* As_expert = As_batched.data() + current_expert * padded_m;
      
      int64_t gathered = 0;
      for (int64_t mb = mb_start; mb < mb_end; ++mb) {
        const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
        int64_t mb_size = offsets[mb + 1] - offsets[mb];
        for (int64_t m = 0; m < mb_size; ++m) {
          int32_t sorted_id = A_ids_mb[m];
          int32_t index = sorted_id / topk;
          copy_stub(A_expert + gathered * K, Aq_tmp + index * K, K);
          As_expert[gathered] = As_tmp[index];
          gathered_ids_per_expert[current_expert][gathered] = sorted_id;
          gathered++;
        }
      }
    }
    
    auto a_mem = memory(a_md, eng, A_batched.data());
    auto c_mem = memory(c_md, eng, C_batched.data());
    
    // Use cached pre-reordered weights (already in optimal 3D format)
    auto& b_mem = w1_cache->expert_weights[0];
    
    // Execute single batched matmul for ALL E experts
    prim.execute(s, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    s.wait();

    // Apply input scales, weight scales, and silu_and_mul for active experts only
    for (const auto& batch : expert_batches) {
      int32_t current_expert = std::get<2>(batch);
      int64_t m_size = expert_m_sizes[current_expert];
      
      const int32_t* Bcomp_w1 = w1_cache->expert_comp[current_expert].data();
      const float* Bs = w1s + current_expert * 2 * N;
      const float* Bs0 = Bs;
      const float* Bs1 = Bs + N;
      const int32_t* Bcomp0 = Bcomp_w1;
      const int32_t* Bcomp1 = Bcomp_w1 + N;
      
      const int32_t* C_expert = C_batched.data() + current_expert * padded_m * 2 * N;
      const float* As_expert = As_batched.data() + current_expert * padded_m;
      const auto& gathered_ids = gathered_ids_per_expert[current_expert];
      
      at::parallel_for(0, m_size, 0, [&](int64_t m_begin, int64_t m_end_local) {
        for (int64_t m = m_begin; m < m_end_local; ++m) {
          float a_scale = As_expert[m];
          const int32_t* C0 = C_expert + m * 2 * N;
          const int32_t* C1 = C_expert + m * 2 * N + N;
          int32_t out_idx = gathered_ids[m];
          
          scale_and_silu_mul_fused<scalar_t>(
              ic1 + out_idx * N,
              C0, C1,
              a_scale,
              Bs0, Bs1,
              Bcomp0, Bcomp1,
              N);
        }
      });
    }
  }

  // stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * N, As_tmp[m], ic1 + m * N, N);
    }
  });

  // ============================================================================
  // Stage 2: Second matmul (w2) - down projection
  // TRUE BATCHED MATMUL: All E experts processed in one batched GEMM call
  // Uses 3D tensors: A[E, padded_m, N] x B[E, N, K] = C[E, padded_m, K]
  // ============================================================================
  if (max_m_size > 0 && num_active_experts > 0) {
    int64_t padded_m = div_up(max_m_size, BLOCK_M) * BLOCK_M;
    
    // Create 3D batched matmul primitive: [E, padded_m, IC] x [E, IC, OC] = [E, padded_m, OC]
    // Use ALL E experts to match the cached weight tensor shape
    auto a_md = memory::desc({E, padded_m, IC}, memory::data_type::u8, memory::format_tag::abc);
    auto c_md = memory::desc({E, padded_m, OC}, memory::data_type::s32, memory::format_tag::abc);

    // Use the cached optimal weight memory descriptor
    matmul::primitive_desc pd(eng, a_md, w2_cache->opt_md, c_md);
    matmul prim(pd);
    
    // Allocate batched buffers for ALL E experts
    std::vector<uint8_t> A_batched(E * padded_m * IC, 0);  // Zero-initialized for inactive experts
    std::vector<int32_t> C_batched(E * padded_m * OC, 0);
    std::vector<float> As_batched(E * padded_m, 0.f);
    std::vector<std::vector<int32_t>> gathered_ids_per_expert(E);
    
    // Gather inputs for active experts only (inactive experts stay zero)
    for (const auto& batch : expert_batches) {
      int64_t mb_start = std::get<0>(batch);
      int64_t mb_end = std::get<1>(batch);
      int32_t current_expert = std::get<2>(batch);
      
      int64_t m_size = expert_m_sizes[current_expert];
      gathered_ids_per_expert[current_expert].resize(m_size);
      
      // Gather input activations from quantized ic1
      uint8_t* A_expert = A_batched.data() + current_expert * padded_m * IC;
      float* As_expert = As_batched.data() + current_expert * padded_m;
      
      int64_t gathered = 0;
      for (int64_t mb = mb_start; mb < mb_end; ++mb) {
        const int32_t* A_ids_mb = sorted_ids + mb * BLOCK_M;
        int64_t mb_size = offsets[mb + 1] - offsets[mb];
        for (int64_t m = 0; m < mb_size; ++m) {
          int32_t sorted_id = A_ids_mb[m];
          copy_stub(A_expert + gathered * IC, Aq_tmp + sorted_id * N, IC);
          As_expert[gathered] = As_tmp[sorted_id];
          gathered_ids_per_expert[current_expert][gathered] = sorted_id;
          gathered++;
        }
      }
    }
    
    auto a_mem = memory(a_md, eng, A_batched.data());
    auto c_mem = memory(c_md, eng, C_batched.data());
    
    // Use cached pre-reordered weights (already in optimal 3D format)
    auto& b_mem = w2_cache->expert_weights[0];
    
    // Execute single batched matmul for ALL E experts
    prim.execute(s, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    s.wait();

    // Apply scales and weights for active experts only
    for (const auto& batch : expert_batches) {
      int32_t current_expert = std::get<2>(batch);
      int64_t m_size = expert_m_sizes[current_expert];
      
      const int32_t* Bcomp_w2 = w2_cache->expert_comp[current_expert].data();
      const float* Bs = w2s + current_expert * K;
      
      const int32_t* C_expert = C_batched.data() + current_expert * padded_m * OC;
      const float* As_expert = As_batched.data() + current_expert * padded_m;
      const auto& gathered_ids = gathered_ids_per_expert[current_expert];
      
      at::parallel_for(0, m_size, 0, [&](int64_t m_begin, int64_t m_end_local) {
        for (int64_t m = m_begin; m < m_end_local; ++m) {
          float a_scale = As_expert[m];
          int32_t index = gathered_ids[m];
          float weight = topk_weights[index];
          const int32_t* C_row = C_expert + m * OC;
          
          scale_and_weight_fused<scalar_t>(
              ic2 + index * K,
              C_row,
              a_scale,
              Bs,
              Bcomp_w2,
              weight,
              OC);
        }
      });
    }
  }

  // stage 3: out = intermediate_cache2.sum(dim=1)
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#endif  // SGLANG_USE_ONEDNN_MOE

#define INSTANTIATE_MOE_INT8_TEMPLATE(TYPE)           \
  template void fused_experts_int8_kernel_impl<TYPE>( \
      TYPE* __restrict__ output,                      \
      TYPE* __restrict__ ic1,                         \
      TYPE* __restrict__ ic2,                         \
      uint8_t* __restrict__ A_tmp,                    \
      float* __restrict__ C_tmp,                      \
      uint8_t* __restrict__ Aq_tmp,                   \
      float* __restrict__ As_tmp,                     \
      const TYPE* __restrict__ input,                 \
      const int8_t* __restrict__ packed_w1,           \
      const int8_t* __restrict__ packed_w2,           \
      const float* __restrict__ w1s,                  \
      const float* __restrict__ w2s,                  \
      const float* __restrict__ topk_weights,         \
      const int32_t* __restrict__ sorted_ids,         \
      const int32_t* __restrict__ expert_ids,         \
      const int32_t* __restrict__ offsets,            \
      int64_t M,                                      \
      int64_t N,                                      \
      int64_t K,                                      \
      int64_t E,                                      \
      int64_t topk,                                   \
      int64_t num_tokens_post_pad)

INSTANTIATE_MOE_INT8_TEMPLATE(at::BFloat16);
INSTANTIATE_MOE_INT8_TEMPLATE(at::Half);

template <typename scalar_t>
void shared_expert_int8_kernel_impl(
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

  const bool use_brgemm = can_use_brgemm<int8_t>(M);

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

      if (use_brgemm) {
        // 1.b gemm: C0 = A @ B0
        at::native::cpublas::brgemm(
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
        at::native::cpublas::brgemm(
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
      } else {
        // fused 1.bcd: silu_and_mul(A @ B0, A @ B1)
        tinygemm_kernel(
            /* A     */ A,
            /* B0    */ B0,
            /* B1    */ B1,
            /* C     */ ic1 + mb * BLOCK_M * N + nb * BLOCK_N,
            /* As    */ As,
            /* Bs0   */ Bs0,
            /* Bs1   */ Bs1,
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ K,
            /* lda   */ K,
            /* ldb   */ n_size,
            /* ldc   */ N);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
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

      if (use_brgemm) {
        at::native::cpublas::brgemm(
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
      } else {
        // 2.a gemm: C = A @ B
        tinygemm_kernel<scalar_t>(
            /* A     */ A,
            /* B     */ B,
            /* C     */ C,
            /* As    */ As,
            /* Bs    */ Bs,
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ IC,
            /* lda   */ IC,
            /* ldb   */ n_size,
            /* ldc   */ BLOCK_N);
      }

      // 2.b copy from C to output and add fused_experts_out
      scalar_t* __restrict__ out = output + mb * BLOCK_M * K + nb * BLOCK_N;
      const scalar_t* __restrict__ fused_out = fused_experts_out + mb * BLOCK_M * K + nb * BLOCK_N;
      for (int64_t m = 0; m < m_size; ++m) {
        add_mul_stub(out + m * K, C + m * BLOCK_N, fused_out + m * K, routed_scaling_factor, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

#define INSTANTIATE_SHARED_EXPERT_INT8_TEMPLATE(TYPE) \
  template void shared_expert_int8_kernel_impl<TYPE>( \
      TYPE* __restrict__ output,                      \
      TYPE* __restrict__ ic1,                         \
      float* __restrict__ C_tmp,                      \
      uint8_t* __restrict__ Aq_tmp,                   \
      float* __restrict__ As_tmp,                     \
      const TYPE* __restrict__ input,                 \
      const int8_t* __restrict__ packed_w1,           \
      const int8_t* __restrict__ packed_w2,           \
      const float* __restrict__ w1s,                  \
      const float* __restrict__ w2s,                  \
      const TYPE* __restrict__ fused_experts_out,     \
      float routed_scaling_factor,                    \
      int64_t M,                                      \
      int64_t N,                                      \
      int64_t K)

INSTANTIATE_SHARED_EXPERT_INT8_TEMPLATE(at::BFloat16);
INSTANTIATE_SHARED_EXPERT_INT8_TEMPLATE(at::Half);
