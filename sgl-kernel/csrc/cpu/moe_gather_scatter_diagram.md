# MoE Gather/Scatter Operations Diagram

## Overview

In the MoE (Mixture of Experts) kernel, tokens are routed to different experts based on a gating mechanism. 
The gather operation collects tokens destined for each expert, and scatter writes results back.

## Example Setup

```
M = 8 tokens, E = 3 experts, TopK = 1
Token routing: tokens 0,3,5 → Expert 0
               tokens 1,4,7 → Expert 1  
               tokens 2,6   → Expert 2
```

## Visual Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT TOKENS (M=8)                                 │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                              │
│  │ T0  │ T1  │ T2  │ T3  │ T4  │ T5  │ T6  │ T7  │   Original layout in memory │
│  │ E0  │ E1  │ E2  │ E0  │ E1  │ E0  │ E2  │ E1  │   (each token assigned to   │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘    one expert)               │
│     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  sorted_ids = [0, 3, 5, 1, 4, 7, 2, 6]
                                    │  expert_ids = [0, 0, 0, 1, 1, 1, 2, 2]
                                    │  offsets    = [0, 3, 6, 8]  (cumulative token counts per expert block)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          GATHER OPERATION                                       │
│                                                                                 │
│  For each expert, collect its assigned tokens into contiguous memory:          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Expert 0 Buffer:  ┌─────┬─────┬─────┐                                   │   │
│  │                   │ T0  │ T3  │ T5  │  ← memcpy from original positions │   │
│  │                   └─────┴─────┴─────┘                                   │   │
│  │                                                                         │   │
│  │ Expert 1 Buffer:  ┌─────┬─────┬─────┐                                   │   │
│  │                   │ T1  │ T4  │ T7  │  ← memcpy from original positions │   │
│  │                   └─────┴─────┴─────┘                                   │   │
│  │                                                                         │   │
│  │ Expert 2 Buffer:  ┌─────┬─────┐                                         │   │
│  │                   │ T2  │ T6  │      ← memcpy from original positions   │   │
│  │                   └─────┴─────┘                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Memory overhead: Each token (size K bytes) is copied once                      │
│  For M=384, K=7168: 384 × 7168 = 2.75 MB of data movement!                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MATMUL COMPUTATION                                     │
│                                                                                 │
│  Expert 0: [3 × K] × [K × 2N] = [3 × 2N]  (w1 matmul)                          │
│  Expert 1: [3 × K] × [K × 2N] = [3 × 2N]  (w1 matmul)                          │
│  Expert 2: [2 × K] × [K × 2N] = [2 × 2N]  (w1 matmul)                          │
│                                                                                 │
│  ↓ Apply SiLU activation and element-wise multiply                             │
│                                                                                 │
│  Expert 0: [3 × N] × [N × K] = [3 × K]    (w2 matmul)                          │
│  Expert 1: [3 × N] × [N × K] = [3 × K]    (w2 matmul)                          │
│  Expert 2: [2 × N] × [N × K] = [2 × K]    (w2 matmul)                          │
│                                                                                 │
│  ⏱️ This is the fast part (~0.6ms for oneDNN)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SCATTER OPERATION                                      │
│                                                                                 │
│  Write results back to original token positions:                                │
│                                                                                 │
│  Expert 0 Results:   Expert 1 Results:   Expert 2 Results:                     │
│  ┌─────┬─────┬─────┐ ┌─────┬─────┬─────┐ ┌─────┬─────┐                         │
│  │ R0  │ R3  │ R5  │ │ R1  │ R4  │ R7  │ │ R2  │ R6  │                         │
│  └──┬──┴──┬──┴──┬──┘ └──┬──┴──┬──┴──┬──┘ └──┬──┴──┬──┘                         │
│     │     │     │       │     │     │       │     │                             │
│     ▼     ▼     ▼       ▼     ▼     ▼       ▼     ▼                             │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐                              │
│  │ R0  │ R1  │ R2  │ R3  │ R4  │ R5  │ R6  │ R7  │   Output in original order  │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘                              │
│                                                                                 │
│  Memory overhead: Each result (size K bytes) is written to scattered positions │
└─────────────────────────────────────────────────────────────────────────────────┘


## Comparison: Baseline vs oneDNN Approach

┌─────────────────────────────────────────────────────────────────────────────────┐
│                     BASELINE (brgemm) - Tiled Approach                          │
│                                                                                 │
│  Uses parallel_2d to tile on both M and N dimensions:                          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │  For each (M_block, N_block) tile:                              │           │
│  │    - Process only tokens in this M_block for current expert     │           │
│  │    - Copy tokens from Aq_tmp to local A buffer (BLOCK_M × K)   │           │
│  │    - Compute small GEMM: [BLOCK_M × K] × [K × BLOCK_N]         │           │
│  │    - Write directly to ic1 output via offsets (implicit scatter)│           │
│  │                                                                 │           │
│  │  ⚠️ Still has per-block gather (copy_stub per token)            │           │
│  │  ✅ Better cache locality (small tiles fit in L2)               │           │
│  │  ✅ Weights pre-packed in VNNI format, used directly            │           │
│  │  ✅ Output written to sorted order (ic1 + offset * N)           │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                 │
│  Key code in baseline (Stage 1):                                               │
│  ```cpp                                                                         │
│  for (int64_t m = 0; m < m_size; ++m) {                                        │
│    int32_t index = A_ids[m] / topk;                                            │
│    copy_stub(A + m * K, Aq_tmp + index * K, K);  // Per-block gather           │
│    As[m] = As_tmp[index];                                                       │
│  }                                                                              │
│  // GEMM with brgemm/tinygemm                                                   │
│  // Write to: ic1 + offset * N + nb * BLOCK_N  (sorted order)                  │
│  ```                                                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                     oneDNN Approach (SGLANG_USE_ONEDNN_MOE=1)                   │
│                                                                                 │
│  Must gather all tokens for each expert before matmul:                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │  Step 1: Gather ALL tokens for expert into contiguous buffer    │           │
│  │          - Check contiguity to potentially skip gather          │           │
│  │          - Parallel gather for large token counts (>64)         │           │
│  │  Step 2: Execute large matmul on gathered data                  │           │
│  │          - oneDNN matmul primitive (uses AMX if available)      │           │
│  │  Step 3: Scatter results back to original positions             │           │
│  │          - Fused with scale_and_silu_mul / scale_and_weight     │           │
│  │          - Contiguous output optimization when possible         │           │
│  │                                                                 │           │
│  │  ⚠️ Full data copy required when non-contiguous                 │           │
│  │  ⚠️ Weight unpacking from VNNI to oneDNN format (cached)        │           │
│  │  ✅ Larger matmul can better utilize AMX                        │           │
│  │  ✅ Zero-copy optimization when tokens are contiguous           │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                 │
│  Key optimizations in oneDNN approach:                                         │
│  - Static weight cache (reorder once, reuse across calls)                      │
│  - Memory object reuse (set_data_handle instead of recreate)                   │
│  - Contiguity check for zero-copy gather/scatter                               │
│  - Pre-allocated buffers outside expert loop                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


## Data Flow Comparison

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          BASELINE DATA FLOW                                     │
│                                                                                 │
│  Stage 0: input [M, K] → quantize → Aq_tmp [M, K] (uint8)                      │
│                                                                                 │
│  Stage 1: For each (M_block, N_block):                                         │
│           Aq_tmp[sorted_ids] → local A → brgemm → ic1[offset*N + nb*BLOCK_N]   │
│                                                                                 │
│  Stage 1.5: ic1 [M*topk, N] → quantize → Aq_tmp [M*topk, N]                    │
│                                                                                 │
│  Stage 2: For each (M_block, N_block):                                         │
│           Aq_tmp[sorted] → local A → brgemm → C_tmp → ic2[sorted_id * K]       │
│                                                                                 │
│  Stage 3: ic2 [M, topk, K] → sum_stub → output [M, K]                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          oneDNN DATA FLOW                                       │
│                                                                                 │
│  Stage 0: input [M, K] → quantize → Aq_tmp [M, K] (uint8)                      │
│                                                                                 │
│  Stage 1: For each expert_batch:                                               │
│           Aq_tmp[indices] → A_padded → oneDNN matmul → C_combined              │
│           → scale_and_silu_mul_fused → ic1[out_idx * N]                         │
│                                                                                 │
│  Stage 1.5: ic1 [M*topk, N] → quantize → Aq_tmp [M*topk, N]                    │
│                                                                                 │
│  Stage 2: For each expert_batch:                                               │
│           Aq_tmp[sorted_ids] → A_padded → oneDNN matmul → C_s32                │
│           → scale_and_weight_fused → ic2[index * K]                             │
│                                                                                 │
│  Stage 3: ic2 [M, topk, K] → sum_stub → output [M, K]                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


## Why Gather/Scatter is Needed

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  BASELINE:                                                                      │
│  - Uses BLOCK_M × BLOCK_N tiling, processes small chunks at a time             │
│  - "Gather" is implicit: copy_stub copies K bytes per token to local buffer    │
│  - "Scatter" is implicit: writes to ic1 + offset * N (pre-sorted order)        │
│  - Each thread has its own small local buffer (A_tmp + tid * BLOCK_M * K)      │
│                                                                                 │
│  oneDNN (SGLANG=1):                                                             │
│  - Processes entire expert batch at once (all tokens for one expert)           │
│  - oneDNN matmul requires contiguous input matrix A[m_size, K]                 │
│  - Must gather scattered tokens into contiguous A_padded buffer                 │
│  - Must scatter results back to original token positions in ic1/ic2            │
│                                                                                 │
│  KEY DIFFERENCE:                                                                │
│  - Baseline: per-block gather (small), writes to sorted output directly        │
│  - oneDNN:   per-expert gather (large), then scatter to original positions     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


## Time Breakdown (M=384, N=2048, K=7168, E=1, TopK=1)

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  Baseline (brgemm):     ████████████████████ 0.95 ms                           │
│                         └── Tiled computation with implicit gather/scatter ──┘  │
│                                                                                 │
│  oneDNN (SGLANG=1):     ████████████████████████████████████████ 2.01 ms       │
│                         ├── Gather: ~0.7 ms ──┤                                 │
│                         ├── Matmul: ~0.6 ms ──┤                                 │
│                         ├── Scatter + Post: ~0.7 ms ──┤                         │
│                                                                                 │
│  Overhead ratio:        ~70% time spent on memory operations!                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


## Conclusion

The gather/scatter overhead is significant because:

1. **Data Movement**: For each token, copy K bytes (7168 bytes) to/from expert buffers
2. **Memory Bandwidth**: Total data moved ≈ 2 × M × K + 2 × M × N bytes
3. **Cache Pollution**: Large copies evict useful data from cache
4. **No Fusion**: Cannot fuse gather with oneDNN matmul computation

The baseline mitigates this by:
- Using tiled parallelism (smaller working set fits in cache)
- Writing to sorted output directly (avoiding explicit scatter)
- Processing with pre-packed VNNI weights (no unpacking needed)

Note: Both approaches have gather operations. The baseline's gather is smaller
(per-block, BLOCK_M tokens at a time), while oneDNN's gather is larger 
(per-expert, all tokens for that expert at once).
