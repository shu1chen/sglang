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

// moe_int8_onednn.cpp -- MoE INT8 (u8*s8) using oneDNN batched matmul

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

// Quantize BF16 -> u8 with +128 offset
static void quantize_bf16_to_u8(
    const at::BFloat16* src, uint8_t* dst, float* sc,
    int64_t M, int64_t K) {
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
}

// Gather by sorted_ids (topk-flattened)
static void gather_u8(
    const uint8_t* in, uint8_t* out,
    const float* isc, float* osc,
    const int32_t* ids, int64_t n, int64_t K, int64_t topk) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    int32_t id = ids[i];
    if (id < 0) {
      std::memset(out + i * K, 0, K);
      osc[i] = 0;
    } else {
      int32_t orig = id / topk;
      std::memcpy(out + i * K, in + orig * K, K);
      osc[i] = isc[orig];
    }
  }
}

// Stage 1 post: (s32 - Bcomp) * As * Bs -> SiLU*Mul -> BF16 ic1
static void dequant_silu_mul_to_bf16(
    const int32_t* gemm_out, at::BFloat16* ic1,
    const float* src_sc, const float* w1_scales,
    const int32_t* gate_bcomp, const int32_t* up_bcomp,
    int64_t ntp, int64_t N, int64_t E, int64_t Me) {
  uint16_t* out = reinterpret_cast<uint16_t*>(ic1);
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < ntp; i++) {
    int64_t e = i / Me;
    if (e >= E) e = E - 1;
    float ss = src_sc[i];
    const int32_t* row = gemm_out + i * 2 * N;
    const float* gws = w1_scales + e * 2 * N;
    const float* uws = w1_scales + e * 2 * N + N;
    const int32_t* gbc = gate_bcomp + e * N;
    const int32_t* ubc = up_bcomp + e * N;
    uint16_t* o = out + i * N;
    int64_t n = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 vss = _mm512_set1_ps(ss);
    __m512 v1 = _mm512_set1_ps(1.f);
    for (; n + 16 <= N; n += 16) {
      __m512 g = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + n), _mm512_loadu_si512(gbc + n))),
          vss), _mm512_loadu_ps(gws + n));
      __m512 u = _mm512_mul_ps(_mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(
              _mm512_loadu_si512(row + N + n), _mm512_loadu_si512(ubc + n))),
          vss), _mm512_loadu_ps(uws + n));
      __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), g);
      __m512 sig = _mm512_div_ps(v1, _mm512_add_ps(v1, _mm512_fast_exp_ps(neg)));
      __m512 res = _mm512_mul_ps(_mm512_mul_ps(g, sig), u);
      __m512i ri = _mm512_castps_si512(res);
      __m512i bias = _mm512_and_si512(_mm512_srli_epi32(ri, 16), _mm512_set1_epi32(1));
      ri = _mm512_add_epi32(_mm512_add_epi32(ri, _mm512_set1_epi32(0x7FFF)), bias);
      _mm256_storeu_si256((__m256i*)(o + n),
          _mm512_cvtepi32_epi16(_mm512_srli_epi32(ri, 16)));
    }
#endif
    for (; n < N; n++) {
      float g = ss * gws[n] * (float)(row[n] - gbc[n]);
      float u = ss * uws[n] * (float)(row[N + n] - ubc[n]);
      float silu = g / (1.f + fast_expf(-g));
      float res = silu * u;
      uint32_t rb;
      std::memcpy(&rb, &res, 4);
      o[n] = (uint16_t)((rb + 0x7FFF + ((rb >> 16) & 1)) >> 16);
    }
  }
}

// Stage 2 post: (s32 - Bcomp) * As * Bs * topk_w -> BF16 ic2
static void dequant_copymul_bf16(
    const int32_t* gemm_out, at::BFloat16* ic2,
    const float* src_sc, const float* wsc,
    const int32_t* bcomp, const int32_t* sorted_ids,
    const float* topk_weights, const int32_t* expert_offsets,
    int64_t K, int64_t E) {
  #pragma omp parallel
  {
    for (int64_t e = 0; e < E; e++) {
      int64_t off = expert_offsets[e], me = expert_offsets[e + 1] - off;
      const float* ws = wsc + e * K;
      const int32_t* bc = bcomp + e * K;
      #pragma omp for schedule(static)
      for (int64_t i = 0; i < me; i++) {
        int32_t index = sorted_ids[off + i];
        if (index < 0) continue;
        float ss_tw = src_sc[off + i] * topk_weights[index];
        const int32_t* row = gemm_out + (off + i) * K;
        uint16_t* d = reinterpret_cast<uint16_t*>(ic2 + index * K);
        int64_t k = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512 vst = _mm512_set1_ps(ss_tw);
        for (; k + 16 <= K; k += 16) {
          __m512 val = _mm512_mul_ps(_mm512_mul_ps(
              _mm512_cvtepi32_ps(_mm512_sub_epi32(
                  _mm512_loadu_si512(row + k), _mm512_loadu_si512(bc + k))),
              vst), _mm512_loadu_ps(ws + k));
          __m512i ri = _mm512_castps_si512(val);
          __m512i bias = _mm512_and_si512(_mm512_srli_epi32(ri, 16), _mm512_set1_epi32(1));
          ri = _mm512_add_epi32(_mm512_add_epi32(ri, _mm512_set1_epi32(0x7FFF)), bias);
          _mm256_storeu_si256((__m256i*)(d + k),
              _mm512_cvtepi32_epi16(_mm512_srli_epi32(ri, 16)));
        }
#endif
        for (; k < K; k++) {
          float val = ss_tw * ws[k] * (float)(row[k] - bc[k]);
          uint32_t rb;
          std::memcpy(&rb, &val, 4);
          d[k] = (uint16_t)((rb + 0x7FFF + ((rb >> 16) & 1)) >> 16);
        }
      }
    }
  }
}

// sum BF16 ic2 [M, topk, K] -> BF16 output [M, K]
static void sum_topk_bf16(
    at::BFloat16* out, const at::BFloat16* ic2,
    int64_t M, int64_t K, int64_t topk) {
  #pragma omp parallel for schedule(static)
  for (int64_t m = 0; m < M; m++) {
    const at::BFloat16* src = ic2 + m * topk * K;
    at::BFloat16* dst = out + m * K;
    if (topk == 1) {
      std::memcpy(dst, src, K * sizeof(at::BFloat16));
      continue;
    }
    for (int64_t k = 0; k < K; k++) {
      float sum = 0;
      for (int64_t t = 0; t < topk; t++)
        sum += (float)src[t * K + k];
      dst[k] = static_cast<at::BFloat16>(sum);
    }
  }
}

// Cached oneDNN primitives
struct OneDNNMoEContext {
  dnnl::engine eng;
  dnnl::stream strm;
  dnnl::matmul mm1, mm2;
  dnnl::matmul::primitive_desc pd1, pd2;
  dnnl::memory sp1, sp2;
  dnnl::memory w1_mem, w2_mem;
  at::Tensor gate_bcomp_t, up_bcomp_t, w2_bcomp_t;
  int64_t M = -1, K = -1, N = -1, E = -1, topk = -1, Me = -1, ntp = -1;
  bool initialized = false;
};

static OneDNNMoEContext& get_context() {
  static OneDNNMoEContext ctx;
  return ctx;
}

static void ensure_context(
    int64_t M, int64_t K, int64_t N, int64_t E, int64_t topk,
    const int8_t* w1_data, const int8_t* w2_data) {
  auto& ctx = get_context();
  int64_t tt = M * topk;
  int64_t Me = std::max((int64_t)1, tt / E);
  int64_t ntp = Me * E;

  if (ctx.initialized && ctx.M == M && ctx.K == K && ctx.N == N &&
      ctx.E == E && ctx.topk == topk) {
    return;
  }

  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;

  ctx.eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
  ctx.strm = dnnl::stream(ctx.eng);

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto s1md = dnnl::memory::desc({E, Me, K}, dt::u8, tag::abc);
  auto w1md = dnnl::memory::desc({E, K, 2 * N}, dt::s8, tag::any);
  auto d1md = dnnl::memory::desc({E, Me, 2 * N}, dt::s32, tag::abc);
  ctx.pd1 = dnnl::matmul::primitive_desc(ctx.eng, s1md, w1md, d1md, attr);
  ctx.mm1 = dnnl::matmul(ctx.pd1);
  ctx.sp1 = dnnl::memory(ctx.pd1.scratchpad_desc(), ctx.eng);

  auto s2md = dnnl::memory::desc({E, Me, N}, dt::u8, tag::abc);
  auto w2md = dnnl::memory::desc({E, N, K}, dt::s8, tag::any);
  auto d2md = dnnl::memory::desc({E, Me, K}, dt::s32, tag::abc);
  ctx.pd2 = dnnl::matmul::primitive_desc(ctx.eng, s2md, w2md, d2md, attr);
  ctx.mm2 = dnnl::matmul(ctx.pd2);
  ctx.sp2 = dnnl::memory(ctx.pd2.scratchpad_desc(), ctx.eng);

  // Reorder weights
  auto w1u = dnnl::memory(
      dnnl::memory::desc({E, K, 2 * N}, dt::s8, tag::abc),
      ctx.eng, const_cast<int8_t*>(w1_data));
  ctx.w1_mem = dnnl::memory(ctx.pd1.weights_desc(), ctx.eng);
  dnnl::reorder(w1u, ctx.w1_mem).execute(ctx.strm, w1u, ctx.w1_mem);

  auto w2u = dnnl::memory(
      dnnl::memory::desc({E, N, K}, dt::s8, tag::abc),
      ctx.eng, const_cast<int8_t*>(w2_data));
  ctx.w2_mem = dnnl::memory(ctx.pd2.weights_desc(), ctx.eng);
  dnnl::reorder(w2u, ctx.w2_mem).execute(ctx.strm, w2u, ctx.w2_mem);
  ctx.strm.wait();

  // Compute Bcomp
  ctx.gate_bcomp_t = at::empty({E, N}, at::kInt);
  ctx.up_bcomp_t = at::empty({E, N}, at::kInt);
  ctx.w2_bcomp_t = at::empty({E, K}, at::kInt);

  for (int64_t e = 0; e < E; e++) {
    const int8_t* wp = w1_data + e * K * 2 * N;
    int32_t* gbc = ctx.gate_bcomp_t.data_ptr<int32_t>() + e * N;
    int32_t* ubc = ctx.up_bcomp_t.data_ptr<int32_t>() + e * N;
    for (int64_t n = 0; n < N; n++) {
      int32_t gs = 0, us = 0;
      for (int64_t k = 0; k < K; k++) {
        gs += wp[k * 2 * N + n];
        us += wp[k * 2 * N + N + n];
      }
      gbc[n] = 128 * gs;
      ubc[n] = 128 * us;
    }
    const int8_t* w2p = w2_data + e * N * K;
    int32_t* w2bc = ctx.w2_bcomp_t.data_ptr<int32_t>() + e * K;
    for (int64_t k = 0; k < K; k++) {
      int32_t s = 0;
      for (int64_t n = 0; n < N; n++) s += w2p[n * K + k];
      w2bc[k] = 128 * s;
    }
  }

  ctx.M = M; ctx.K = K; ctx.N = N; ctx.E = E; ctx.topk = topk;
  ctx.Me = Me; ctx.ntp = ntp;
  ctx.initialized = true;
}

}  // anonymous namespace

// Public API
at::Tensor fused_experts_onednn_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,            // [E, K, 2*N] s8
    at::Tensor& w2,            // [E, N, K] s8
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    const at::Tensor& w1_scale,
    const at::Tensor& w2_scale) {
  RECORD_FUNCTION("sgl-kernel::fused_experts_onednn_cpu",
      std::vector<c10::IValue>({hidden_states, w1, w2, topk_weights, topk_ids}));

  TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16,
      "fused_experts_onednn_cpu: only BFloat16 input supported");
  TORCH_CHECK(w1.scalar_type() == at::kChar, "w1 must be int8");
  TORCH_CHECK(w2.scalar_type() == at::kChar, "w2 must be int8");

  CHECK_DIM(2, hidden_states);
  CHECK_DIM(3, w1);
  CHECK_DIM(3, w2);

  int64_t M = hidden_states.size(0);
  int64_t K = hidden_states.size(1);
  int64_t N = w1.size(2) / 2;
  int64_t E = w1.size(0);
  int64_t topk = topk_ids.size(1);

  auto topk_weights_f = topk_weights.to(at::kFloat);

  int64_t tt = M * topk;
  int64_t Me = std::max((int64_t)1, tt / E);
  int64_t ntp = Me * E;

  ensure_context(M, K, N, E, topk,
                 w1.data_ptr<int8_t>(), w2.data_ptr<int8_t>());
  auto& ctx = get_context();

  // Routing
  std::vector<int32_t> sids(ntp);
  std::vector<int32_t> eoffs(E + 1);
  {
    auto* tid_ptr = topk_ids.data_ptr<int32_t>();
    std::vector<std::vector<int32_t>> pe(E);
    for (int64_t t = 0; t < M; t++)
      for (int64_t k = 0; k < topk; k++) {
        int pair_idx = t * topk + k;
        int e_id = tid_ptr[t * topk + k];
        pe[e_id].push_back(pair_idx);
      }
    int o = 0;
    for (int64_t e = 0; e < E; e++) {
      eoffs[e] = o;
      for (int64_t i = 0; i < Me; i++)
        sids[o + i] = (i < (int64_t)pe[e].size()) ? pe[e][i] : -1;
      o += Me;
    }
    eoffs[E] = o;
  }

  float* tw = topk_weights_f.data_ptr<float>();
  float* w1s = w1_scale.data_ptr<float>();
  float* w2s = w2_scale.data_ptr<float>();
  auto inp = hidden_states.data_ptr<at::BFloat16>();

  // Buffers
  auto qi_t = at::empty({M, K}, at::kByte);
  auto isc_t = at::empty({M}, at::kFloat);
  auto ic1_t = at::zeros({ntp, N}, at::kBFloat16);
  auto ic2_t = at::zeros({(int64_t)(M * topk * K)}, at::kBFloat16);
  auto out_t = at::zeros({M, K}, at::kBFloat16);
  auto ssc_t = at::empty({ntp}, at::kFloat);
  auto intsc_t = at::empty({ntp}, at::kFloat);

  uint8_t* qi = qi_t.data_ptr<uint8_t>();
  float* isc = isc_t.data_ptr<float>();
  auto* ic1 = ic1_t.data_ptr<at::BFloat16>();
  auto* ic2 = ic2_t.data_ptr<at::BFloat16>();
  auto* outp = out_t.data_ptr<at::BFloat16>();
  float* ssc = ssc_t.data_ptr<float>();
  float* intsc = intsc_t.data_ptr<float>();

  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;

  auto s1md = dnnl::memory::desc({E, Me, K}, dt::u8, tag::abc);
  auto d1md = dnnl::memory::desc({E, Me, 2 * N}, dt::s32, tag::abc);
  auto s2md = dnnl::memory::desc({E, Me, N}, dt::u8, tag::abc);
  auto d2md = dnnl::memory::desc({E, Me, K}, dt::s32, tag::abc);

  auto s1m = dnnl::memory(s1md, ctx.eng);
  auto d1m = dnnl::memory(d1md, ctx.eng);
  auto s2m = dnnl::memory(s2md, ctx.eng);
  auto d2m = dnnl::memory(d2md, ctx.eng);

  uint8_t* sa = (uint8_t*)s1m.get_data_handle();
  int32_t* d1 = (int32_t*)d1m.get_data_handle();
  uint8_t* qi2 = (uint8_t*)s2m.get_data_handle();
  int32_t* d2 = (int32_t*)d2m.get_data_handle();

  // Stage 0: Quantize input
  quantize_bf16_to_u8(inp, qi, isc, M, K);

  // Gather per-expert
  gather_u8(qi, sa, isc, ssc, sids.data(), ntp, K, topk);

  // Stage 1a: GEMM w1
  std::unordered_map<int, dnnl::memory> a1 = {
      {DNNL_ARG_SRC, s1m}, {DNNL_ARG_WEIGHTS, ctx.w1_mem},
      {DNNL_ARG_DST, d1m}, {DNNL_ARG_SCRATCHPAD, ctx.sp1}};
  ctx.mm1.execute(ctx.strm, a1);
  ctx.strm.wait();

  // Stage 1b: dequant + SiLU*Mul -> BF16 ic1
  dequant_silu_mul_to_bf16(d1, ic1, ssc, w1s,
      ctx.gate_bcomp_t.data_ptr<int32_t>(),
      ctx.up_bcomp_t.data_ptr<int32_t>(),
      ntp, N, E, Me);

  // Stage 1.5: Quantize ic1 -> u8
  quantize_bf16_to_u8(ic1, qi2, intsc, ntp, N);

  // Stage 2a: GEMM w2
  std::unordered_map<int, dnnl::memory> a2 = {
      {DNNL_ARG_SRC, s2m}, {DNNL_ARG_WEIGHTS, ctx.w2_mem},
      {DNNL_ARG_DST, d2m}, {DNNL_ARG_SCRATCHPAD, ctx.sp2}};
  ctx.mm2.execute(ctx.strm, a2);
  ctx.strm.wait();

  // Stage 2b: dequant + copymul -> BF16 ic2
  dequant_copymul_bf16(d2, ic2, intsc, w2s,
      ctx.w2_bcomp_t.data_ptr<int32_t>(),
      sids.data(), tw, eoffs.data(), K, E);

  // Stage 3: sum topk -> BF16 output
  sum_topk_bf16(outp, ic2, M, K, topk);

  return out_t;
}
