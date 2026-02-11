#include "common.h"
#include "gemm.h"
#include "vec.h"

#ifdef SGLANG_USE_ONEDNN_MOE
#include "oneapi/dnnl/dnnl.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <mutex>
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

// Strided version of silu_and_mul: reads C0/C1 from a large [BLOCK_M, ldc] buffer
// where C0 starts at column offset col0 and C1 at col1, both with stride ldc.
template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul_strided(
    scalar_t* __restrict__ C,
    const int32_t* __restrict__ C_full,  // full [BLOCK_M, ldc] buffer
    int64_t col0,                        // column offset for gate (C0)
    int64_t col1,                        // column offset for up (C1)
    int64_t ldc,                         // leading dimension (total columns)
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
    __m512i vc32_0 = _mm512_loadu_si512(C_full + m * ldc + col0 + col * 16);
    __m512i vc32_1 = _mm512_loadu_si512(C_full + m * ldc + col1 + col * 16);
    vc0[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32_0, vcomp0[col]));
    vc1[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32_1, vcomp1[col]));
    vc0[col] = _mm512_mul_ps(_mm512_mul_ps(vc0[col], vas), vbs0[col]);
    vc1[col] = _mm512_mul_ps(_mm512_mul_ps(vc1[col], vas), vbs1[col]);
  };

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  auto silu_and_mul_op = [&](auto col) {
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
    Unroll<COLS>{}(silu_and_mul_op);
    Unroll<COLS>{}(storec, m);
  }
#else
  TORCH_CHECK(false, "silu_and_mul_strided: scalar path not implemented!");
#endif
}

// Strided version of scale_C: reads from a large [BLOCK_M, ldc] buffer at column offset col_off
template <int BLOCK_N>
inline void scale_C_strided(
    float* __restrict__ C,
    const int32_t* __restrict__ C_full,  // full [BLOCK_M, ldc] buffer
    int64_t col_off,                     // column offset
    int64_t ldc,                         // leading dimension
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
    __m512i vc32 = _mm512_loadu_si512(C_full + m * ldc + col_off + col * 16);
    vc[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32, vcomp[col]));
    vc[col] = _mm512_mul_ps(_mm512_mul_ps(vc[col], vas), vbs[col]);
    _mm512_storeu_ps(C + m * BLOCK_N + col * 16, vc[col]);
  };

  for (int64_t m = 0; m < m_size; ++m) {
    Unroll<COLS>{}(scalec, m);
  }
#else
  TORCH_CHECK(false, "scale_C_strided: scalar path not implemented!");
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
// Optimized oneDNN-based implementation
// Uses oneDNN matmul with pre-reordered weights, user-managed scratchpad
// for thread safety, and parallel_2d/loop_2d for cache-friendly threading.
//
// Key optimizations:
// - Weights are unpacked from VNNI and reordered with format_tag::any once,
//   then cached globally keyed by source pointer.
// - C API (dnnl_primitive_execute) with stack-allocated args avoids
//   unordered_map and shared_ptr overhead per iteration.
// - Flat lookup tables for O(1) weight entry access in hot loop.
// - Thread-local stream/memory handles cached to avoid per-parallel_2d
//   object creation overhead.
// - Per-thread scratchpad with scratchpad_mode::user for thread safety.
// ============================================================================

namespace {

// Global oneDNN engine - shared by all threads
inline dnnl::engine& get_onednn_engine() {
  static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  return eng;
}

// Unpacks VNNI format weights to plain [K, N] format
inline void unpack_vnni_weights(
    int8_t* __restrict__ dst,
    const int8_t* __restrict__ src,
    int64_t K,
    int64_t N) {
  constexpr int VNNI_BLK = 4;
  const int64_t K_groups = K / VNNI_BLK;
  for (int64_t k_group = 0; k_group < K_groups; ++k_group) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t d = 0; d < VNNI_BLK; ++d) {
        int64_t k = k_group * VNNI_BLK + d;
        dst[k * N + n] = src[k_group * N * VNNI_BLK + n * VNNI_BLK + d];
      }
    }
  }
}

// ---- Global weight cache keyed by source pointer ----
struct OneDNNWeightCache {
  struct CacheEntry {
    dnnl::memory reordered_weight;   // owns the memory (shared_ptr lifetime)
    dnnl_memory_t raw_weight;        // raw handle for fast C API execute
    dnnl_primitive_t raw_primitive;   // raw handle for fast C API execute
    dnnl::matmul::primitive_desc prim_desc;
    dnnl::matmul primitive;          // owns the primitive (shared_ptr lifetime)
    dnnl::memory::desc scratchpad_md;
  };

  std::unordered_map<const void*, CacheEntry> entries;
  std::mutex mutex;

  static OneDNNWeightCache& instance() {
    static OneDNNWeightCache cache;
    return cache;
  }
};

// Pre-reorder a single VNNI weight block [K_dim, N_dim] and cache it.
inline const OneDNNWeightCache::CacheEntry* get_or_create_cached_weight(
    const int8_t* weight_ptr,
    int64_t K_dim,
    int64_t N_dim,
    int64_t BLOCK_M_dim
) {
  auto& cache = OneDNNWeightCache::instance();

  // Lock-free fast path
  {
    auto it = cache.entries.find(weight_ptr);
    if (it != cache.entries.end()) {
      return &it->second;
    }
  }

  auto& eng = get_onednn_engine();
  dnnl::stream s(eng);

  dnnl::memory::desc a_md({BLOCK_M_dim, K_dim}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_user_md({K_dim, N_dim}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_opt_md({K_dim, N_dim}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
  dnnl::memory::desc c_md({BLOCK_M_dim, N_dim}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ab);

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_opt_md, c_md, attr);
  auto prim = dnnl::matmul(pd);
  auto sp_md = pd.scratchpad_desc();

  // Unpack VNNI weights to plain format [K, N]
  std::vector<int8_t> unpacked(K_dim * N_dim);
  unpack_vnni_weights(unpacked.data(), weight_ptr, K_dim, N_dim);

  auto b_user_mem = dnnl::memory(b_user_md, eng, unpacked.data());
  dnnl::memory b_reordered;
  if (pd.weights_desc() != b_user_md) {
    b_reordered = dnnl::memory(pd.weights_desc(), eng);
    dnnl::reorder(b_user_mem, b_reordered).execute(s, b_user_mem, b_reordered);
    s.wait();
  } else {
    b_reordered = dnnl::memory(b_user_md, eng);
    std::memcpy(b_reordered.get_data_handle(), unpacked.data(), K_dim * N_dim);
  }

  // Capture raw handles before moving
  dnnl_memory_t raw_w = b_reordered.get();
  dnnl_primitive_t raw_p = prim.get();

  std::lock_guard<std::mutex> lock(cache.mutex);
  auto [it, inserted] = cache.entries.emplace(
      weight_ptr,
      OneDNNWeightCache::CacheEntry{
          std::move(b_reordered), raw_w, raw_p,
          std::move(pd), std::move(prim), std::move(sp_md)});
  return &it->second;
}

// Create a paired gate+up weight block [K, 2*BLOCK_N] from two separate
// VNNI blocks and cache it. The key is a synthetic pointer: (gate_ptr).
// This halves the number of dnnl_primitive_execute calls in Stage 1.
inline const OneDNNWeightCache::CacheEntry* get_or_create_paired_weight(
    const int8_t* gate_ptr,
    const int8_t* up_ptr,
    int64_t K_dim,
    int64_t BLOCK_N_dim,
    int64_t BLOCK_M_dim
) {
  // Use gate_ptr as key (unique per expert+nb pair)
  const void* key = reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(gate_ptr) | 1ULL);  // tag bit to distinguish from single-block entries

  auto& cache = OneDNNWeightCache::instance();

  {
    auto it = cache.entries.find(key);
    if (it != cache.entries.end()) {
      return &it->second;
    }
  }

  auto& eng = get_onednn_engine();
  dnnl::stream s(eng);

  const int64_t N2 = 2 * BLOCK_N_dim;
  dnnl::memory::desc a_md({BLOCK_M_dim, K_dim}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_user_md({K_dim, N2}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_opt_md({K_dim, N2}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
  dnnl::memory::desc c_md({BLOCK_M_dim, N2}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ab);

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_opt_md, c_md, attr);
  auto prim = dnnl::matmul(pd);
  auto sp_md = pd.scratchpad_desc();

  // Unpack and interleave gate[K,BLOCK_N] and up[K,BLOCK_N] into [K, 2*BLOCK_N]
  std::vector<int8_t> unpacked(K_dim * N2);
  // Unpack gate to columns [0, BLOCK_N)
  {
    constexpr int VNNI_BLK = 4;
    const int64_t K_groups = K_dim / VNNI_BLK;
    for (int64_t k_group = 0; k_group < K_groups; ++k_group) {
      for (int64_t n = 0; n < BLOCK_N_dim; ++n) {
        for (int64_t d = 0; d < VNNI_BLK; ++d) {
          int64_t k = k_group * VNNI_BLK + d;
          unpacked[k * N2 + n] = gate_ptr[k_group * BLOCK_N_dim * VNNI_BLK + n * VNNI_BLK + d];
        }
      }
    }
  }
  // Unpack up to columns [BLOCK_N, 2*BLOCK_N)
  {
    constexpr int VNNI_BLK = 4;
    const int64_t K_groups = K_dim / VNNI_BLK;
    for (int64_t k_group = 0; k_group < K_groups; ++k_group) {
      for (int64_t n = 0; n < BLOCK_N_dim; ++n) {
        for (int64_t d = 0; d < VNNI_BLK; ++d) {
          int64_t k = k_group * VNNI_BLK + d;
          unpacked[k * N2 + BLOCK_N_dim + n] = up_ptr[k_group * BLOCK_N_dim * VNNI_BLK + n * VNNI_BLK + d];
        }
      }
    }
  }

  auto b_user_mem = dnnl::memory(b_user_md, eng, unpacked.data());
  dnnl::memory b_reordered;
  if (pd.weights_desc() != b_user_md) {
    b_reordered = dnnl::memory(pd.weights_desc(), eng);
    dnnl::reorder(b_user_mem, b_reordered).execute(s, b_user_mem, b_reordered);
    s.wait();
  } else {
    b_reordered = dnnl::memory(b_user_md, eng);
    std::memcpy(b_reordered.get_data_handle(), unpacked.data(), K_dim * N2);
  }

  dnnl_memory_t raw_w = b_reordered.get();
  dnnl_primitive_t raw_p = prim.get();

  std::lock_guard<std::mutex> lock(cache.mutex);
  auto [it, inserted] = cache.entries.emplace(
      key,
      OneDNNWeightCache::CacheEntry{
          std::move(b_reordered), raw_w, raw_p,
          std::move(pd), std::move(prim), std::move(sp_md)});
  return &it->second;
}

// ---- Global flat lookup table cache ----
// Caches the flat entry tables keyed by (packed_w_ptr, E, NB, K, N, BLOCK_M, BLOCK_N)
// to avoid per-call std::vector allocation and E*NB hash lookups.
struct FlatTableCache {
  struct TableKey {
    const void* weight_ptr;
    int64_t E, NB, K_dim, N_dim, BLOCK_M_dim;
    bool operator==(const TableKey& o) const {
      return weight_ptr == o.weight_ptr && E == o.E && NB == o.NB &&
             K_dim == o.K_dim && N_dim == o.N_dim && BLOCK_M_dim == o.BLOCK_M_dim;
    }
  };
  struct TableKeyHash {
    size_t operator()(const TableKey& k) const {
      return std::hash<const void*>{}(k.weight_ptr) ^
             (std::hash<int64_t>{}(k.E) << 1) ^
             (std::hash<int64_t>{}(k.NB) << 2);
    }
  };

  struct W1Table {
    // w1_paired[e * NB + nb] = paired gate+up entry for [K, 2*BLOCK_N]
    std::vector<const OneDNNWeightCache::CacheEntry*> paired_entries;
  };
  struct W2Table {
    // w2_entries[e * NB2 + nb]
    std::vector<const OneDNNWeightCache::CacheEntry*> entries;
  };

  std::unordered_map<TableKey, W1Table, TableKeyHash> w1_tables;
  std::unordered_map<TableKey, W2Table, TableKeyHash> w2_tables;
  std::mutex mutex;

  static FlatTableCache& instance() {
    static FlatTableCache cache;
    return cache;
  }

  // Get or build w1 paired flat table — each entry is [K, 2*BLOCK_N] gate+up pair
  const std::vector<const OneDNNWeightCache::CacheEntry*>& get_w1_paired_entries(
      const int8_t* packed_w1, int64_t E, int64_t NB, int64_t K, int64_t N,
      int64_t BLOCK_N, int64_t BLOCK_M, int64_t stride_e, int64_t stride_n) {
    TableKey key{packed_w1, E, NB, K, N, BLOCK_M};
    {
      auto it = w1_tables.find(key);
      if (it != w1_tables.end()) return it->second.paired_entries;
    }
    // Build paired table
    std::vector<const OneDNNWeightCache::CacheEntry*> entries(E * NB);
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t nb = 0; nb < NB; ++nb) {
        const int8_t* p_gate = packed_w1 + e * stride_e + nb * BLOCK_N * stride_n;
        const int8_t* p_up = packed_w1 + e * stride_e + (NB + nb) * BLOCK_N * stride_n;
        entries[e * NB + nb] = get_or_create_paired_weight(p_gate, p_up, K, BLOCK_N, BLOCK_M);
      }
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto [it, _] = w1_tables.emplace(key, W1Table{std::move(entries)});
    return it->second.paired_entries;
  }

  // Get or build w2 flat table
  const std::vector<const OneDNNWeightCache::CacheEntry*>& get_w2_entries(
      const int8_t* packed_w2, int64_t E, int64_t NB2, int64_t IC, int64_t OC,
      int64_t BLOCK_N, int64_t BLOCK_M, int64_t stride_e2, int64_t stride_oc) {
    TableKey key{packed_w2, E, NB2, IC, OC, BLOCK_M};
    {
      auto it = w2_tables.find(key);
      if (it != w2_tables.end()) return it->second.entries;
    }
    // Build table
    std::vector<const OneDNNWeightCache::CacheEntry*> entries(E * NB2);
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t nb = 0; nb < NB2; ++nb) {
        const int8_t* p = packed_w2 + e * stride_e2 + nb * BLOCK_N * stride_oc;
        entries[e * NB2 + nb] = get_or_create_cached_weight(p, IC, BLOCK_N, BLOCK_M);
      }
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto [it, _] = w2_tables.emplace(key, W2Table{std::move(entries)});
    return it->second.entries;
  }
};

// ---- Thread-local dnnl handle cache ----
// Avoids creating dnnl::stream, dnnl::memory objects (shared_ptr overhead)
// on every parallel_2d invocation. Handles are created once per thread and
// resized/reconfigured as needed.
struct ThreadLocalHandles {
  // Stream
  dnnl::stream stream;
  dnnl_stream_t raw_stream;

  // Stage 1 handles (paired gate+up: output is [BLOCK_M, 2*BLOCK_N])
  dnnl::memory a1_mem;       // [BLOCK_M, K]
  dnnl::memory c1_mem;       // [BLOCK_M, 2*BLOCK_N]
  dnnl::memory sp1_mem;      // scratchpad
  dnnl_memory_t raw_a1;
  dnnl_memory_t raw_c1;
  dnnl_memory_t raw_sp1;

  // Stage 2 handles
  dnnl::memory a2_mem;       // [BLOCK_M, IC]
  dnnl::memory c2_mem;       // [BLOCK_M, BLOCK_N]
  dnnl::memory sp2_mem;      // scratchpad
  dnnl_memory_t raw_a2;
  dnnl_memory_t raw_c2;
  dnnl_memory_t raw_sp2;

  // Config tracking to avoid re-creation
  int64_t configured_K = 0;
  int64_t configured_IC = 0;
  int64_t configured_BLOCK_N = 0;
  int64_t configured_BLOCK_M = 0;
  size_t configured_sp1_size = 0;
  size_t configured_sp2_size = 0;

  // Pointer tracking to avoid redundant dnnl_memory_set_data_handle calls
  void* s1_a_ptr = nullptr;
  void* s1_c_ptr = nullptr;
  void* s2_a_ptr = nullptr;
  void* s2_c_ptr = nullptr;

  bool initialized = false;

  void ensure_stage1(dnnl::engine& eng, int64_t BLOCK_M_val, int64_t K_val,
                     int64_t BLOCK_N_val, const dnnl::memory::desc& sp_md,
                     uint8_t* A_buf, int32_t* C_buf) {
    using namespace dnnl;
    size_t sp_size = sp_md.get_size();
    if (!initialized || configured_K != K_val || configured_BLOCK_M != BLOCK_M_val ||
        configured_BLOCK_N != BLOCK_N_val || configured_sp1_size != sp_size) {
      stream = dnnl::stream(eng);
      raw_stream = stream.get();

      memory::desc a_md({BLOCK_M_val, K_val}, memory::data_type::u8, memory::format_tag::ab);
      memory::desc c_md({BLOCK_M_val, 2 * BLOCK_N_val}, memory::data_type::s32, memory::format_tag::ab);
      a1_mem = memory(a_md, eng, A_buf);
      c1_mem = memory(c_md, eng, C_buf);
      raw_a1 = a1_mem.get();
      raw_c1 = c1_mem.get();

      if (sp_size > 0) {
        sp1_mem = memory(sp_md, eng);
      }
      raw_sp1 = sp1_mem.get(true);

      configured_K = K_val;
      configured_BLOCK_M = BLOCK_M_val;
      configured_BLOCK_N = BLOCK_N_val;
      configured_sp1_size = sp_size;
      initialized = true;
      s1_a_ptr = A_buf;
      s1_c_ptr = C_buf;
    } else if (s1_a_ptr != A_buf || s1_c_ptr != C_buf) {
      // Only update data pointers when they actually change
      dnnl_memory_set_data_handle(raw_a1, A_buf);
      dnnl_memory_set_data_handle(raw_c1, C_buf);
      s1_a_ptr = A_buf;
      s1_c_ptr = C_buf;
    }
    // else: same config + same pointers → no-op (fast path)
  }

  void ensure_stage2(dnnl::engine& eng, int64_t BLOCK_M_val, int64_t IC_val,
                     int64_t BLOCK_N_val, const dnnl::memory::desc& sp_md,
                     uint8_t* A_buf, int32_t* C_buf) {
    using namespace dnnl;
    size_t sp_size = sp_md.get_size();
    if (!initialized || configured_IC != IC_val ||
        configured_sp2_size != sp_size) {
      if (!initialized) {
        stream = dnnl::stream(eng);
        raw_stream = stream.get();
        initialized = true;
      }

      memory::desc a_md({BLOCK_M_val, IC_val}, memory::data_type::u8, memory::format_tag::ab);
      memory::desc c_md({BLOCK_M_val, BLOCK_N_val}, memory::data_type::s32, memory::format_tag::ab);
      a2_mem = memory(a_md, eng, A_buf);
      c2_mem = memory(c_md, eng, C_buf);
      raw_a2 = a2_mem.get();
      raw_c2 = c2_mem.get();

      if (sp_size > 0) {
        sp2_mem = memory(sp_md, eng);
      }
      raw_sp2 = sp2_mem.get(true);

      configured_IC = IC_val;
      configured_sp2_size = sp_size;
      s2_a_ptr = A_buf;
      s2_c_ptr = C_buf;
    } else if (s2_c_ptr != C_buf) {
      // A pointer changes in inner loop via dnnl_memory_set_data_handle
      // C pointer is stable per thread — only update if changed
      dnnl_memory_set_data_handle(raw_c2, C_buf);
      s2_c_ptr = C_buf;
    }
    // else: same config + same pointers → no-op (fast path)
  }
};

// Thread-local storage for handles — avoids per-parallel_2d creation overhead
inline ThreadLocalHandles& get_thread_handles() {
  static thread_local ThreadLocalHandles handles;
  return handles;
}

}  // anonymous namespace

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

  // Stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * K, As_tmp[m], input + m * K, K);
    }
  });

  // Stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);
  const int64_t stride_e = 2 * N * packed_K;
  const int64_t stride_n = packed_K;

  auto& eng = get_onednn_engine();

  // Use globally cached flat lookup tables — avoids per-call vector allocation
  // and E*NB hash lookups. Tables are built once and reused.
  // Stage 1 uses paired gate+up entries: one [K, 2*BLOCK_N] matmul per (mb,nb)
  // instead of two [K, BLOCK_N] matmuls, halving primitive execute calls.
  auto& ftcache = FlatTableCache::instance();
  const auto& w1_paired = ftcache.get_w1_paired_entries(
      packed_w1, E, NB, K, N, BLOCK_N, BLOCK_M, stride_e, stride_n);

  // Get a sample entry for scratchpad desc and raw data pointer for fast access
  const auto* w1_sample = w1_paired[0];
  const auto* const* w1_data = w1_paired.data();

  // Stage 1: w1 matmul using parallel_2d + loop_2d
  // Single paired matmul [BLOCK_M, K] x [K, 2*BLOCK_N] -> [BLOCK_M, 2*BLOCK_N]
  // per (mb, nb) iteration, followed by silu_and_mul_strided.
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    uint8_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    // C buffer holds [BLOCK_M, 2*BLOCK_N] int32 results
    int32_t* __restrict__ C_paired = reinterpret_cast<int32_t*>(C_tmp) + tid * 2 * BLOCK_M * BLOCK_N;

    alignas(64) float As[BLOCK_M];

    // Use thread-local cached handles to avoid creating dnnl objects each time
    auto& th = get_thread_handles();
    th.ensure_stage1(eng, BLOCK_M, K, BLOCK_N, w1_sample->scratchpad_md, A, C_paired);

    // Pre-build exec_args template — only weight pointer changes per iteration
    dnnl_exec_arg_t exec_args[] = {
        {DNNL_ARG_SRC, th.raw_a1},
        {DNNL_ARG_WEIGHTS, nullptr},  // filled per iteration
        {DNNL_ARG_DST, th.raw_c1},
        {DNNL_ARG_SCRATCHPAD, th.raw_sp1}
    };
    dnnl_stream_t raw_s = th.raw_stream;

    int64_t last_mb = -1;
    int64_t cur_m_size = 0;
    loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K * 2, [&](int64_t mb, int64_t nb, int64_t) {
      int32_t expert_id = expert_ids[mb];

      const int8_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb * BLOCK_N * stride_n;
      const int8_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + (NB + nb) * BLOCK_N * stride_n;
      const float* __restrict__ Bs0 = w1s + expert_id * 2 * N + nb * BLOCK_N;
      const float* __restrict__ Bs1 = w1s + expert_id * 2 * N + N + nb * BLOCK_N;
      const int32_t* Bcomp0 = reinterpret_cast<const int32_t*>(B0 + BLOCK_N * K);
      const int32_t* Bcomp1 = reinterpret_cast<const int32_t*>(B1 + BLOCK_N * K);

      if (mb != last_mb) {
        last_mb = mb;
        const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
        cur_m_size = offsets[mb + 1] - offsets[mb];

        if (cur_m_size < BLOCK_M) {
          std::memset(A + cur_m_size * K, 0, (BLOCK_M - cur_m_size) * K);
        }
        for (int64_t m = 0; m < cur_m_size; ++m) {
          int32_t index = A_ids[m] / topk;
          copy_stub(A + m * K, Aq_tmp + index * K, K);
          As[m] = As_tmp[index];
        }
      }

      // Single paired matmul: [BLOCK_M, K] x [K, 2*BLOCK_N] -> [BLOCK_M, 2*BLOCK_N]
      const int64_t w1_idx = expert_id * NB + nb;
      exec_args[1].memory = w1_data[w1_idx]->raw_weight;
      dnnl_primitive_execute(w1_data[w1_idx]->raw_primitive, raw_s, 4, exec_args);

      // silu_and_mul from strided [BLOCK_M, 2*BLOCK_N] buffer
      // gate at columns [0, BLOCK_N), up at columns [BLOCK_N, 2*BLOCK_N)
      const int64_t offset = offsets[mb];
      const int64_t ldc = 2 * BLOCK_N;
      silu_and_mul_strided<scalar_t, BLOCK_N>(
          ic1 + offset * N + nb * BLOCK_N, C_paired, 0, BLOCK_N, ldc,
          As, Bs0, Bs1, Bcomp0, Bcomp1, cur_m_size, N);
    });
  });

  // Stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * N, As_tmp[m], ic1 + m * N, N);
    }
  });

  // Stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * packed_N;
  const int64_t stride_oc = packed_N;

  // Use globally cached flat lookup table for w2
  const auto& w2_entries = ftcache.get_w2_entries(
      packed_w2, E, NB2, IC, OC, BLOCK_N, BLOCK_M, stride_e2, stride_oc);
  const auto* w2_sample = w2_entries[0];
  const auto* const* w2_data = w2_entries.data();

  // Pre-allocate per-thread A padding buffers for Stage 2 to avoid
  // heap allocation (std::vector) inside parallel_2d.
  // Need BLOCK_M * IC bytes per thread. If A_tmp per-thread buffer (BLOCK_M * K)
  // is large enough (K >= IC), reuse it; otherwise allocate once.
  const bool need_a2_pad = (K < IC);
  std::vector<uint8_t> a2_pad_pool;
  int num_threads = 1;
#if defined(_OPENMP)
  num_threads = omp_get_max_threads();
#endif
  if (need_a2_pad) {
    a2_pad_pool.resize(num_threads * BLOCK_M * IC, 0);
  }

  // Stage 2: w2 matmul using parallel_2d + loop_2d (C API execution)
  parallel_2d(MB, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    int32_t* __restrict__ C32 = reinterpret_cast<int32_t*>(C + BLOCK_M * BLOCK_N);

    uint8_t* __restrict__ A_pad = need_a2_pad
        ? (a2_pad_pool.data() + tid * BLOCK_M * IC)
        : (A_tmp + tid * BLOCK_M * K);

    // Use thread-local cached handles
    auto& th = get_thread_handles();
    th.ensure_stage2(eng, BLOCK_M, IC, BLOCK_N, w2_sample->scratchpad_md, A_pad, C32);

    // Pre-build exec_args template — only weight and src pointers change
    dnnl_exec_arg_t exec_args[] = {
        {DNNL_ARG_SRC, th.raw_a2},
        {DNNL_ARG_WEIGHTS, nullptr},  // filled per iteration
        {DNNL_ARG_DST, th.raw_c2},
        {DNNL_ARG_SCRATCHPAD, th.raw_sp2}
    };
    dnnl_stream_t raw_s = th.raw_stream;
    dnnl_memory_t raw_a = th.raw_a2;

    int64_t last_mb = -1;  // track last mb to skip redundant A setup

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      const uint8_t* __restrict__ A_orig = Aq_tmp + offsets[mb] * N;
      const float* __restrict__ As = As_tmp + offsets[mb];
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      int32_t expert_id = expert_ids[mb];
      const int8_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const float* __restrict__ Bs = w2s + expert_id * K + nb * BLOCK_N;
      const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + BLOCK_N * IC);

      if (mb != last_mb) {
        last_mb = mb;
        if (m_size < BLOCK_M) {
          // Need to pad — copy + zero-fill
          std::memcpy(A_pad, A_orig, m_size * IC);
          std::memset(A_pad + m_size * IC, 0, (BLOCK_M - m_size) * IC);
          dnnl_memory_set_data_handle(raw_a, A_pad);
        } else {
          // m_size == BLOCK_M — use A_orig directly, no copy needed
          dnnl_memory_set_data_handle(raw_a, const_cast<uint8_t*>(A_orig));
        }
      }

      const int64_t w2_idx = expert_id * NB2 + nb;
      exec_args[1].memory = w2_data[w2_idx]->raw_weight;

      dnnl_primitive_execute(w2_data[w2_idx]->raw_primitive, raw_s, 4, exec_args);

      scale_C<BLOCK_N>(C, C32, As, Bs, Bcomp, m_size);

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    });
  });

  // Stage 3: out = intermediate_cache2.sum(dim=1)
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
