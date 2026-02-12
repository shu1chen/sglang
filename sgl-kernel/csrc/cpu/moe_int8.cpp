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
// Strided version of silu_and_mul: reads gate from C0_buf at col0 offset
// and up from C1_buf at col1 offset, both with leading dim ldc.
template <typename scalar_t, int BLOCK_N>
inline void silu_and_mul_strided(
    scalar_t* __restrict__ C,
    const int32_t* __restrict__ C0_buf,  // gate buffer [BLOCK_M, ldc]
    int64_t col0,                        // column offset within C0_buf
    const int32_t* __restrict__ C1_buf,  // up buffer [BLOCK_M, ldc]
    int64_t col1,                        // column offset within C1_buf
    const float* __restrict__ As,
    const float* __restrict__ Bs0,
    const float* __restrict__ Bs1,
    const int32_t* __restrict__ Bcomp0,
    const int32_t* __restrict__ Bcomp1,
    int64_t m_size,
    int64_t N,
    int64_t ldc) {
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
    __m512i vc32_0 = _mm512_loadu_si512(C0_buf + m * ldc + col0 + col * 16);
    __m512i vc32_1 = _mm512_loadu_si512(C1_buf + m * ldc + col1 + col * 16);
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

// Strided version of scale_C: reads from C_in at col_in offset with stride ldc_in,
// writes to C_out at col_out offset with stride ldc_out
template <int BLOCK_N>
inline void scale_C_strided(
    float* __restrict__ C_out,
    int64_t col_out,                     // column offset in output
    const int32_t* __restrict__ C_in,    // input [BLOCK_M, ldc_in] buffer
    int64_t col_in,                      // column offset in input
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    const int32_t* __restrict__ Bcomp,
    int64_t m_size,
    int64_t ldc) {
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
    __m512i vc32 = _mm512_loadu_si512(C_in + m * ldc + col_in + col * 16);
    vc[col] = _mm512_cvtepi32_ps(_mm512_sub_epi32(vc32, vcomp[col]));
    vc[col] = _mm512_mul_ps(_mm512_mul_ps(vc[col], vas), vbs[col]);
    _mm512_storeu_ps(C_out + m * ldc + col_out + col * 16, vc[col]);
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
// oneDNN matmul-based implementation — per-block with parallel_2d/loop_2d
//
// Uses the same cache-friendly blocking as baseline (parallel_2d + loop_2d)
// but replaces brgemm with oneDNN matmul primitives. Each weight block
// [K, MATMUL_N] is cached as an optimally-formatted oneDNN memory with
// format_tag::any, and the matmul primitive is also cached.
//
// MATMUL_N (128) is wider than BLOCK_N (32) to reduce the number of
// dnnl_primitive_execute calls. Post-processing (silu_and_mul, scale_C)
// still operates on BLOCK_N=32 sub-blocks since Bcomp/Bs are per-BLOCK_N.
//
// Key optimizations:
//   - Wide matmul blocks: 4× fewer dispatch calls (MATMUL_N/BLOCK_N = 4)
//   - Global weight cache: one reorder per weight block, amortized over calls
//   - Thread-local stream/memory: avoid per-call dnnl object creation
//   - Flat lookup table: O(1) expert×block → CacheEntry* via raw array
//   - C API execution: dnnl_primitive_execute with stack-allocated args
//   - last_mb caching: skip redundant A copies when same mb repeats
//   - Pointer tracking: skip redundant dnnl_memory_set_data_handle
// ============================================================================

// Width of oneDNN matmul blocks — must be multiple of BLOCK_N
static constexpr int64_t MATMUL_N = matmul_block_n();

namespace {

// Global oneDNN engine - shared by all threads
inline dnnl::engine& get_onednn_engine() {
  static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  return eng;
}

// ---- Global weight cache: keyed by raw pointer of VNNI block ----
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

// Unpack VNNI [K/4, BLOCK_N, 4] to plain [K, BLOCK_N] and create oneDNN cached entry
// For a single BLOCK_N-wide block.
inline const OneDNNWeightCache::CacheEntry* get_or_create_weight(
    const int8_t* vnni_ptr,
    int64_t K_dim,
    int64_t N_dim,
    int64_t BLOCK_M_dim) {
  auto& cache = OneDNNWeightCache::instance();
  const void* key = vnni_ptr;

  // Lock-free fast path
  {
    auto it = cache.entries.find(key);
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

  // Unpack from VNNI to plain [K, N]
  std::vector<int8_t> unpacked(K_dim * N_dim);
  constexpr int VNNI_BLK = 4;
  const int64_t K_groups = K_dim / VNNI_BLK;
  for (int64_t k_group = 0; k_group < K_groups; ++k_group) {
    for (int64_t n = 0; n < N_dim; ++n) {
      for (int64_t d = 0; d < VNNI_BLK; ++d) {
        int64_t k = k_group * VNNI_BLK + d;
        unpacked[k * N_dim + n] = vnni_ptr[k_group * N_dim * VNNI_BLK + n * VNNI_BLK + d];
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
    std::memcpy(b_reordered.get_data_handle(), unpacked.data(), K_dim * N_dim);
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

// Create a wide matmul entry by concatenating num_sub adjacent VNNI blocks
// into a single [K, wide_N] matrix. Each sub-block is [K/4, BLOCK_N_sub, 4]
// with stride stride_n between sub-blocks.
// Key is the pointer to the first sub-block.
inline const OneDNNWeightCache::CacheEntry* get_or_create_wide_weight(
    const int8_t* first_vnni_ptr,
    int64_t K_dim,
    int64_t BLOCK_N_sub,
    int64_t num_sub,
    int64_t stride_n,  // byte stride between sub-blocks (packed_K)
    int64_t BLOCK_M_dim) {
  auto& cache = OneDNNWeightCache::instance();
  // Use a unique key: combine first_vnni_ptr with num_sub to distinguish wide vs narrow
  // Since wide entries start at same ptr as narrow, we offset the key
  const void* key = reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(first_vnni_ptr) | (static_cast<uintptr_t>(num_sub) << 48));

  // Lock-free fast path
  {
    auto it = cache.entries.find(key);
    if (it != cache.entries.end()) {
      return &it->second;
    }
  }

  const int64_t wide_N = BLOCK_N_sub * num_sub;
  auto& eng = get_onednn_engine();
  dnnl::stream s(eng);

  dnnl::memory::desc a_md({BLOCK_M_dim, K_dim}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_user_md({K_dim, wide_N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_opt_md({K_dim, wide_N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
  dnnl::memory::desc c_md({BLOCK_M_dim, wide_N}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ab);

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_opt_md, c_md, attr);
  auto prim = dnnl::matmul(pd);
  auto sp_md = pd.scratchpad_desc();

  // Unpack from multiple VNNI blocks to plain [K, wide_N]
  std::vector<int8_t> unpacked(K_dim * wide_N);
  constexpr int VNNI_BLK = 4;
  const int64_t K_groups = K_dim / VNNI_BLK;

  for (int64_t s_idx = 0; s_idx < num_sub; ++s_idx) {
    const int8_t* vnni_ptr = first_vnni_ptr + s_idx * BLOCK_N_sub * stride_n;
    const int64_t col_offset = s_idx * BLOCK_N_sub;
    for (int64_t k_group = 0; k_group < K_groups; ++k_group) {
      for (int64_t n = 0; n < BLOCK_N_sub; ++n) {
        for (int64_t d = 0; d < VNNI_BLK; ++d) {
          int64_t k = k_group * VNNI_BLK + d;
          unpacked[k * wide_N + col_offset + n] =
              vnni_ptr[k_group * BLOCK_N_sub * VNNI_BLK + n * VNNI_BLK + d];
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
    std::memcpy(b_reordered.get_data_handle(), unpacked.data(), K_dim * wide_N);
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

// Create a fused gate+up matmul: concatenate gate and up VNNI blocks side-by-side
// into a single [K, 2*wide_n] matrix.
//   gate_ptr points to the first gate VNNI sub-block
//   up_ptr   points to the first up   VNNI sub-block
//   num_sub VNNI sub-blocks per half, stride_n between sub-blocks
// Output matmul: [BLOCK_M, K] × [K, 2*wide_n] → [BLOCK_M, 2*wide_n]
// Columns [0, wide_n) = gate, columns [wide_n, 2*wide_n) = up
inline const OneDNNWeightCache::CacheEntry* get_or_create_fused_gate_up_weight(
    const int8_t* gate_ptr,
    const int8_t* up_ptr,
    int64_t K_dim,
    int64_t BLOCK_N_sub,
    int64_t num_sub,
    int64_t stride_n,
    int64_t BLOCK_M_dim) {
  auto& cache = OneDNNWeightCache::instance();
  // Unique key: use gate_ptr combined with a marker (bit 47 set) and num_sub in bits 48+
  const void* key = reinterpret_cast<const void*>(
      reinterpret_cast<uintptr_t>(gate_ptr) |
      (1ULL << 47) |
      (static_cast<uintptr_t>(num_sub) << 48));

  {
    auto it = cache.entries.find(key);
    if (it != cache.entries.end()) {
      return &it->second;
    }
  }

  const int64_t half_N = BLOCK_N_sub * num_sub;
  const int64_t fused_N = 2 * half_N;
  auto& eng = get_onednn_engine();
  dnnl::stream s(eng);

  dnnl::memory::desc a_md({BLOCK_M_dim, K_dim}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_user_md({K_dim, fused_N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_opt_md({K_dim, fused_N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
  dnnl::memory::desc c_md({BLOCK_M_dim, fused_N}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ab);

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_opt_md, c_md, attr);
  auto prim = dnnl::matmul(pd);
  auto sp_md = pd.scratchpad_desc();

  // Unpack gate and up VNNI blocks into plain [K, 2*half_N]
  // Columns [0, half_N) = gate, [half_N, 2*half_N) = up
  std::vector<int8_t> unpacked(K_dim * fused_N, 0);
  constexpr int VNNI_BLK = 4;
  const int64_t K_groups = K_dim / VNNI_BLK;

  // Gate half: columns [0, half_N)
  for (int64_t s_idx = 0; s_idx < num_sub; ++s_idx) {
    const int8_t* vnni = gate_ptr + s_idx * BLOCK_N_sub * stride_n;
    const int64_t col_off = s_idx * BLOCK_N_sub;
    for (int64_t kg = 0; kg < K_groups; ++kg) {
      for (int64_t n = 0; n < BLOCK_N_sub; ++n) {
        for (int64_t d = 0; d < VNNI_BLK; ++d) {
          int64_t k = kg * VNNI_BLK + d;
          unpacked[k * fused_N + col_off + n] =
              vnni[kg * BLOCK_N_sub * VNNI_BLK + n * VNNI_BLK + d];
        }
      }
    }
  }
  // Up half: columns [half_N, 2*half_N)
  for (int64_t s_idx = 0; s_idx < num_sub; ++s_idx) {
    const int8_t* vnni = up_ptr + s_idx * BLOCK_N_sub * stride_n;
    const int64_t col_off = half_N + s_idx * BLOCK_N_sub;
    for (int64_t kg = 0; kg < K_groups; ++kg) {
      for (int64_t n = 0; n < BLOCK_N_sub; ++n) {
        for (int64_t d = 0; d < VNNI_BLK; ++d) {
          int64_t k = kg * VNNI_BLK + d;
          unpacked[k * fused_N + col_off + n] =
              vnni[kg * BLOCK_N_sub * VNNI_BLK + n * VNNI_BLK + d];
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
    std::memcpy(b_reordered.get_data_handle(), unpacked.data(), K_dim * fused_N);
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

// ---- Flat lookup table: entries[expert * NB_wide + nb_wide] ----
// Each entry points to a wide matmul CacheEntry covering MATMUL_N columns.
struct FlatTableCache {
  struct TableKey {
    const void* weight_ptr;
    int64_t total_entries;
    int64_t matmul_n;  // distinguish different widths
    bool operator==(const TableKey& o) const {
      return weight_ptr == o.weight_ptr && total_entries == o.total_entries && matmul_n == o.matmul_n;
    }
  };
  struct TableKeyHash {
    size_t operator()(const TableKey& k) const {
      return std::hash<const void*>{}(k.weight_ptr) ^
             (std::hash<int64_t>{}(k.total_entries) << 1) ^
             (std::hash<int64_t>{}(k.matmul_n) << 2);
    }
  };
  struct Table {
    std::vector<const OneDNNWeightCache::CacheEntry*> entries;
  };

  std::unordered_map<TableKey, Table, TableKeyHash> tables;
  std::mutex mutex;

  static FlatTableCache& instance() {
    static FlatTableCache cache;
    return cache;
  }

  // Build wide flat table for w1: entries[e * 2*NB_wide + nb_wide]
  // First NB_wide entries per expert = gate blocks, next NB_wide = up blocks
  // Each entry covers actual_matmul_n columns (may be < matmul_n for last block)
  const std::vector<const OneDNNWeightCache::CacheEntry*>& get_w1_table(
      const int8_t* packed_w1, int64_t E, int64_t NB, int64_t NB_wide,
      int64_t K, int64_t BLOCK_N_val, int64_t matmul_n, int64_t BLOCK_M,
      int64_t stride_e, int64_t stride_n, int64_t N) {
    const int64_t total = E * 2 * NB_wide;
    TableKey key{packed_w1, total, matmul_n};
    {
      auto it = tables.find(key);
      if (it != tables.end()) return it->second.entries;
    }

    const int64_t sub_per_wide = matmul_n / BLOCK_N_val;
    std::vector<const OneDNNWeightCache::CacheEntry*> entries(total);
    for (int64_t e = 0; e < E; ++e) {
      // Gate and up halves each have NB sub-blocks
      for (int64_t half = 0; half < 2; ++half) {
        for (int64_t nbw = 0; nbw < NB_wide; ++nbw) {
          int64_t first_nb = nbw * sub_per_wide;  // first sub-block index within this half
          int64_t remaining = NB - first_nb;  // remaining sub-blocks
          int64_t num_sub = std::min(sub_per_wide, remaining);
          int64_t abs_nb = half * NB + first_nb;  // index in the full 2*N layout
          const int8_t* ptr = packed_w1 + e * stride_e + abs_nb * BLOCK_N_val * stride_n;
          if (num_sub == sub_per_wide) {
            entries[e * 2 * NB_wide + half * NB_wide + nbw] =
                get_or_create_wide_weight(ptr, K, BLOCK_N_val, num_sub, stride_n, BLOCK_M);
          } else {
            // Last block may be narrower - still create with actual width
            entries[e * 2 * NB_wide + half * NB_wide + nbw] =
                get_or_create_wide_weight(ptr, K, BLOCK_N_val, num_sub, stride_n, BLOCK_M);
          }
        }
      }
    }

    std::lock_guard<std::mutex> lock(mutex);
    auto [it, _] = tables.emplace(key, Table{std::move(entries)});
    return it->second.entries;
  }

  // Build fused gate+up flat table for w1: entries[e * NB_wide + nbw]
  // Each entry is a fused [K, 2*wide_n] matmul covering gate and up together.
  // Output columns: [0, wide_n) = gate, [wide_n, 2*wide_n) = up.
  const std::vector<const OneDNNWeightCache::CacheEntry*>& get_w1_fused_table(
      const int8_t* packed_w1, int64_t E, int64_t NB, int64_t NB_wide,
      int64_t K, int64_t BLOCK_N_val, int64_t matmul_n, int64_t BLOCK_M,
      int64_t stride_e, int64_t stride_n, int64_t N) {
    const int64_t total = E * NB_wide;
    // Use a distinct key by adding 3 to matmul_n to avoid collision with separate tables
    TableKey key{packed_w1, total, matmul_n + 3};
    {
      auto it = tables.find(key);
      if (it != tables.end()) return it->second.entries;
    }

    const int64_t sub_per_wide = matmul_n / BLOCK_N_val;
    std::vector<const OneDNNWeightCache::CacheEntry*> entries(total);
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t nbw = 0; nbw < NB_wide; ++nbw) {
        int64_t first_nb = nbw * sub_per_wide;
        int64_t remaining = NB - first_nb;
        int64_t num_sub = std::min(sub_per_wide, remaining);
        // Gate pointer: half=0
        const int8_t* gate_ptr = packed_w1 + e * stride_e + first_nb * BLOCK_N_val * stride_n;
        // Up pointer: half=1 (offset by NB sub-blocks)
        const int8_t* up_ptr = packed_w1 + e * stride_e + (NB + first_nb) * BLOCK_N_val * stride_n;
        entries[e * NB_wide + nbw] =
            get_or_create_fused_gate_up_weight(gate_ptr, up_ptr, K, BLOCK_N_val, num_sub, stride_n, BLOCK_M);
      }
    }

    std::lock_guard<std::mutex> lock(mutex);
    auto [it, _] = tables.emplace(key, Table{std::move(entries)});
    return it->second.entries;
  }

  // Build wide flat table for w2: entries[e * NB2_wide + nb_wide]
  const std::vector<const OneDNNWeightCache::CacheEntry*>& get_w2_table(
      const int8_t* packed_w2, int64_t E, int64_t NB2, int64_t NB2_wide,
      int64_t IC, int64_t BLOCK_N_val, int64_t matmul_n, int64_t BLOCK_M,
      int64_t stride_e2, int64_t stride_oc, int64_t OC) {
    const int64_t total = E * NB2_wide;
    TableKey key{packed_w2, total, matmul_n};
    {
      auto it = tables.find(key);
      if (it != tables.end()) return it->second.entries;
    }

    const int64_t sub_per_wide = matmul_n / BLOCK_N_val;
    std::vector<const OneDNNWeightCache::CacheEntry*> entries(total);
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t nbw = 0; nbw < NB2_wide; ++nbw) {
        int64_t first_nb = nbw * sub_per_wide;
        int64_t remaining = NB2 - first_nb;
        int64_t num_sub = std::min(sub_per_wide, remaining);
        const int8_t* ptr = packed_w2 + e * stride_e2 + first_nb * BLOCK_N_val * stride_oc;
        entries[e * NB2_wide + nbw] =
            get_or_create_wide_weight(ptr, IC, BLOCK_N_val, num_sub, stride_oc, BLOCK_M);
      }
    }

    std::lock_guard<std::mutex> lock(mutex);
    auto [it, _] = tables.emplace(key, Table{std::move(entries)});
    return it->second.entries;
  }
};

// ---- Thread-local dnnl handle cache ----
struct ThreadLocalHandles {
  dnnl::stream stream;
  dnnl_stream_t raw_stream = nullptr;

  // Stage 1 fused gate+up: [BLOCK_M, K] × [K, 2*matmul_n] → [BLOCK_M, 2*matmul_n]
  dnnl::memory a1_mem, c1_fused_mem, sp1_mem;
  dnnl_memory_t raw_a1 = nullptr, raw_c1_fused = nullptr, raw_sp1 = nullptr;

  // Stage 2 handles: [BLOCK_M, IC] × [IC, matmul_n] → [BLOCK_M, matmul_n]
  dnnl::memory a2_mem, c2_mem, sp2_mem;
  dnnl_memory_t raw_a2 = nullptr, raw_c2 = nullptr, raw_sp2 = nullptr;

  // Config tracking
  int64_t s1_K = 0, s1_N = 0;
  int64_t s2_IC = 0, s2_N = 0;
  size_t s1_sp_size = 0, s2_sp_size = 0;
  bool initialized = false;

  // Pointer tracking to skip redundant set_data_handle
  void* last_a1_ptr = nullptr;
  void* last_c1_fused_ptr = nullptr;
  void* last_a2_ptr = nullptr;
  void* last_c2_ptr = nullptr;

  // Stage 1: fused gate+up output is [BLOCK_M, 2*matmul_n] s32
  void ensure_stage1(dnnl::engine& eng, int64_t BLOCK_M_val, int64_t K_val,
                     int64_t matmul_n_val, const dnnl::memory::desc& sp_md,
                     void* A_buf, int32_t* C_fused_buf) {
    using namespace dnnl;
    const int64_t fused_n = 2 * matmul_n_val;
    size_t sp_size = sp_md.get_size();
    if (!initialized || s1_K != K_val || s1_N != matmul_n_val || s1_sp_size != sp_size) {
      stream = dnnl::stream(eng);
      raw_stream = stream.get();

      memory::desc a_md({BLOCK_M_val, K_val}, memory::data_type::u8, memory::format_tag::ab);
      memory::desc c_md({BLOCK_M_val, fused_n}, memory::data_type::s32, memory::format_tag::ab);
      a1_mem = memory(a_md, eng, A_buf);
      c1_fused_mem = memory(c_md, eng, C_fused_buf);
      raw_a1 = a1_mem.get();
      raw_c1_fused = c1_fused_mem.get();

      if (sp_size > 0) {
        sp1_mem = memory(sp_md, eng);
      }
      raw_sp1 = sp1_mem.get(true);

      s1_K = K_val;
      s1_N = matmul_n_val;
      s1_sp_size = sp_size;
      initialized = true;
      last_a1_ptr = A_buf;
      last_c1_fused_ptr = C_fused_buf;
    } else {
      // Shape unchanged but buffers may have moved (new allocation)
      if (A_buf != last_a1_ptr) {
        dnnl_memory_set_data_handle(raw_a1, A_buf);
        last_a1_ptr = A_buf;
      }
      if (C_fused_buf != last_c1_fused_ptr) {
        dnnl_memory_set_data_handle(raw_c1_fused, C_fused_buf);
        last_c1_fused_ptr = C_fused_buf;
      }
    }
  }

  void ensure_stage2(dnnl::engine& eng, int64_t BLOCK_M_val, int64_t IC_val,
                     int64_t matmul_n_val, const dnnl::memory::desc& sp_md,
                     void* A_buf, int32_t* C_buf) {
    using namespace dnnl;
    size_t sp_size = sp_md.get_size();
    if (s2_IC != IC_val || s2_N != matmul_n_val || s2_sp_size != sp_size) {
      if (!initialized) {
        stream = dnnl::stream(eng);
        raw_stream = stream.get();
        initialized = true;
      }

      memory::desc a_md({BLOCK_M_val, IC_val}, memory::data_type::u8, memory::format_tag::ab);
      memory::desc c_md({BLOCK_M_val, matmul_n_val}, memory::data_type::s32, memory::format_tag::ab);
      a2_mem = memory(a_md, eng, A_buf);
      c2_mem = memory(c_md, eng, C_buf);
      raw_a2 = a2_mem.get();
      raw_c2 = c2_mem.get();

      if (sp_size > 0) {
        sp2_mem = memory(sp_md, eng);
      }
      raw_sp2 = sp2_mem.get(true);

      s2_IC = IC_val;
      s2_N = matmul_n_val;
      s2_sp_size = sp_size;
      last_a2_ptr = A_buf;
      last_c2_ptr = C_buf;
    } else {
      // Shape unchanged but buffers may have moved
      if (A_buf && A_buf != last_a2_ptr) {
        dnnl_memory_set_data_handle(raw_a2, A_buf);
        last_a2_ptr = A_buf;
      }
      if (C_buf != last_c2_ptr) {
        dnnl_memory_set_data_handle(raw_c2, C_buf);
        last_c2_ptr = C_buf;
      }
    }
  }
};

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

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // Stage 0: quantize input to uint8, [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * K, As_tmp[m], input + m * K, K);
    }
  });

  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  TORCH_CHECK(N % BLOCK_N == 0, "Fixme when N is not multiples of ", BLOCK_N);

  const int64_t packed_K = get_row_size<int8_t>(K);
  const int64_t packed_N = get_row_size<int8_t>(N);
  const int64_t stride_e = 2 * N * packed_K;
  const int64_t stride_n = packed_K;

  // Compute wide block parameters
  // actual_matmul_n: the width of the wide matmul, clamped to N if N < MATMUL_N
  const int64_t actual_matmul_n = std::min(MATMUL_N, N);
  TORCH_CHECK(actual_matmul_n % BLOCK_N == 0, "MATMUL_N must be multiple of BLOCK_N");
  TORCH_CHECK(N % actual_matmul_n == 0, "N must be multiple of actual_matmul_n for fused gate+up");
  const int64_t sub_per_wide = actual_matmul_n / BLOCK_N;
  const int64_t NB_wide = N / actual_matmul_n;

  auto& eng = get_onednn_engine();

  // Build wide flat lookup tables
  auto& ftc = FlatTableCache::instance();
  const auto& w1_table = ftc.get_w1_fused_table(
      packed_w1, E, NB, NB_wide, K, BLOCK_N, actual_matmul_n, BLOCK_M,
      stride_e, stride_n, N);
  const auto* const* w1_lut = w1_table.data();

  // Get a sample entry for scratchpad desc
  const auto* sample = w1_lut[0];

  // ---- Stage 1: fused gate+up matmul → silu_and_mul ----
  // One matmul per (mb, nbw): [BLOCK_M, K] × [K, 2*wide_n] → [BLOCK_M, 2*wide_n]
  // Output columns: [0, wide_n) = gate, [wide_n, 2*wide_n) = up
  const int64_t fused_matmul_n = 2 * actual_matmul_n;  // output width of fused matmul
  parallel_2d(MB, NB_wide, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    uint8_t* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    // Fused C buffer: [BLOCK_M, 2*actual_matmul_n] s32
    int32_t* __restrict__ C_fused = reinterpret_cast<int32_t*>(C_tmp) + tid * 2 * BLOCK_M * actual_matmul_n;

    alignas(64) float As[BLOCK_M];

    auto& th = get_thread_handles();
    th.ensure_stage1(eng, BLOCK_M, K, actual_matmul_n, sample->scratchpad_md, A, C_fused);

    dnnl_exec_arg_t exec_args[] = {
        {DNNL_ARG_SRC, th.raw_a1},
        {DNNL_ARG_WEIGHTS, nullptr},
        {DNNL_ARG_DST, th.raw_c1_fused},
        {DNNL_ARG_SCRATCHPAD, th.raw_sp1}
    };
    dnnl_stream_t raw_s = th.raw_stream;

    int64_t last_mb = -1;

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, fused_matmul_n * K, [&](int64_t mb, int64_t nbw, int64_t nb_offset) {
      int32_t expert_id = expert_ids[mb];
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      // Load A block (skip if same mb as last time)
      if (mb != last_mb) {
        const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
        if (m_size < BLOCK_M) {
          std::memset(A + m_size * K, 0, (BLOCK_M - m_size) * K);
        }
        for (int64_t m = 0; m < m_size; ++m) {
          int32_t index = A_ids[m] / topk;
          copy_stub(A + m * K, Aq_tmp + index * K, K);
          As[m] = As_tmp[index];
        }
        last_mb = mb;
      }

      // Number of BLOCK_N sub-blocks in this wide block (may be fewer at the end)
      int64_t first_nb = nbw * sub_per_wide;
      int64_t num_sub = std::min(sub_per_wide, NB - first_nb);
      int64_t wide_n = num_sub * BLOCK_N;

      // Fused gate+up matmul: C_fused = A @ [B_gate | B_up]
      // [BLOCK_M, K] × [K, 2*wide_n] → [BLOCK_M, 2*wide_n]
      const auto* entry = w1_lut[expert_id * NB_wide + nbw];
      exec_args[1].memory = entry->raw_weight;
      dnnl_primitive_execute(entry->raw_primitive, raw_s, 4, exec_args);

      // Dequantize + silu_and_mul for each BLOCK_N sub-block
      // C_fused layout: columns [0, wide_n) = gate, [wide_n, 2*wide_n) = up
      const int64_t offset = offsets[mb];
      for (int64_t s = 0; s < num_sub; ++s) {
        int64_t nb = first_nb + s;
        const float* __restrict__ Bs0 = w1s + expert_id * 2 * N + nb * BLOCK_N;
        const float* __restrict__ Bs1 = w1s + expert_id * 2 * N + (NB + nb) * BLOCK_N;
        const int8_t* __restrict__ B0 = packed_w1 + expert_id * stride_e + nb * BLOCK_N * stride_n;
        const int8_t* __restrict__ B1 = packed_w1 + expert_id * stride_e + (NB + nb) * BLOCK_N * stride_n;
        const int32_t* Bcomp0 = reinterpret_cast<const int32_t*>(B0 + BLOCK_N * K);
        const int32_t* Bcomp1 = reinterpret_cast<const int32_t*>(B1 + BLOCK_N * K);

        // Gate: col offset = s * BLOCK_N (in first half)
        // Up:   col offset = wide_n + s * BLOCK_N (in second half)
        // ldc = 2 * wide_n (fused row width)
        silu_and_mul_strided<scalar_t, BLOCK_N>(
            ic1 + offset * N + nb * BLOCK_N,
            C_fused, s * BLOCK_N, C_fused, wide_n + s * BLOCK_N,
            As, Bs0, Bs1, Bcomp0, Bcomp1, m_size, N, fused_matmul_n);
      }
    });
  });

  // Stage 1.5: quantize ic1 to uint8, [M * topk, N]
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_int8<scalar_t>(Aq_tmp + m * N, As_tmp[m], ic1 + m * N, N);
    }
  });
  // Zero-fill padding rows after valid data so oneDNN matmul reads zeros
  // (oneDNN reads BLOCK_M rows but only m_size are valid)
  {
    int64_t pad_start = M * topk;
    int64_t pad_end = pad_start + BLOCK_M;
    std::memset(Aq_tmp + pad_start * N, 0, BLOCK_M * N);
    for (int64_t m = pad_start; m < pad_end; ++m) {
      As_tmp[m] = 0.0f;
    }
  }

  // ---- Stage 2: intermediate_cache2 = intermediate_cache1 @ w2 ----
  const int64_t OC = K;
  const int64_t IC = N;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_e2 = OC * packed_N;
  const int64_t stride_oc = packed_N;

  const int64_t actual_matmul_n2 = std::min(MATMUL_N, OC);
  const int64_t sub_per_wide2 = actual_matmul_n2 / BLOCK_N;
  const int64_t NB2_wide = div_up(OC, actual_matmul_n2);

  const auto& w2_table = ftc.get_w2_table(
      packed_w2, E, NB2, NB2_wide, IC, BLOCK_N, actual_matmul_n2, BLOCK_M,
      stride_e2, stride_oc, OC);
  const auto* const* w2_lut = w2_table.data();

  const auto* sample2 = w2_lut[0];

  parallel_2d(MB, NB2_wide, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    float* __restrict__ C = C_tmp + tid * 2 * BLOCK_M * actual_matmul_n2;
    int32_t* __restrict__ C32 = reinterpret_cast<int32_t*>(C + BLOCK_M * actual_matmul_n2);

    auto& th = get_thread_handles();
    th.ensure_stage2(eng, BLOCK_M, IC, actual_matmul_n2, sample2->scratchpad_md, nullptr, C32);

    dnnl_exec_arg_t exec_args[] = {
        {DNNL_ARG_SRC, th.raw_a2},
        {DNNL_ARG_WEIGHTS, nullptr},
        {DNNL_ARG_DST, th.raw_c2},
        {DNNL_ARG_SCRATCHPAD, th.raw_sp2}
    };
    dnnl_stream_t raw_s = th.raw_stream;

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, actual_matmul_n2 * IC, [&](int64_t mb, int64_t nbw, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      // A ptr from ic1 quantized, in sorted order
      const uint8_t* __restrict__ A = Aq_tmp + offsets[mb] * N;
      const float* __restrict__ As = As_tmp + offsets[mb];
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      int32_t expert_id = expert_ids[mb];

      // Update A pointer (avoid redundant set_data_handle)
      void* a_ptr = const_cast<uint8_t*>(A);
      if (a_ptr != th.last_a2_ptr) {
        dnnl_memory_set_data_handle(th.raw_a2, a_ptr);
        th.last_a2_ptr = a_ptr;
      }
      exec_args[0].memory = th.raw_a2;

      // Number of BLOCK_N sub-blocks in this wide block
      int64_t first_nb = nbw * sub_per_wide2;
      int64_t num_sub = std::min(sub_per_wide2, NB2 - first_nb);
      int64_t wide_n = num_sub * BLOCK_N;

      // Wide matmul: C32 = A @ B_wide
      const auto* entry = w2_lut[expert_id * NB2_wide + nbw];
      exec_args[1].memory = entry->raw_weight;
      dnnl_primitive_execute(entry->raw_primitive, raw_s, 4, exec_args);

      // Dequantize + copy for each BLOCK_N sub-block
      for (int64_t s = 0; s < num_sub; ++s) {
        int64_t nb = first_nb + s;
        int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

        const int8_t* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
        const float* __restrict__ Bs = w2s + expert_id * K + nb * BLOCK_N;
        const int32_t* Bcomp = reinterpret_cast<const int32_t*>(B + BLOCK_N * IC);

        // scale_C reads from C32 at sub-block offset, writes to C at sub-block offset
        scale_C_strided<BLOCK_N>(C, s * BLOCK_N, C32, s * BLOCK_N, As, Bs, Bcomp, m_size, actual_matmul_n2);

        // Copy to ic2 with topk_weights
        for (int64_t m = 0; m < m_size; ++m) {
          int32_t index = A_ids[m];
          float weight = topk_weights[index];
          copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * actual_matmul_n2 + s * BLOCK_N, weight, n_size);
        }
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
