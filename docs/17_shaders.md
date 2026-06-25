# shader/ - GLSL Compute Shader Specifications

=============================================================================
This file documents what each shader should do, its inputs/outputs,
push constants, and dispatch dimensions.
All shaders live in: shader/*.comp
Compiled to: bin/*.spv
=============================================================================

## Shader Naming Convention

  layernorm.spv      - LayerNorm compute shader
  rmsnorm.spv        - RMSNorm compute shader
  rope.spv          - Rotary Position Embedding shader
  attention.spv     - Naive O(n^2) attention shader
  attention_flash.spv - Flash Attention shader
  mlp_gate_up.spv   - Fused gate+up projection + SiLU
  mlp_down.spv      - Down projection
  softmax.spv        - Standalone softmax (usually fused)
  embed.spv         - Embedding lookup shader
  quantize.spv      - FP16 -> INT8 quantization
  dequantize.spv    - INT8 -> FP16 dequantization

## Binding Convention (consistent across all shaders)

  All shaders use the same descriptor set layout:

  layout(set=0, binding=0) readonly  buffer InputBuffer  { float data[]; } in0;
  layout(set=0, binding=1) writeonly buffer OutputBuffer { float data[]; } out0;
  layout(set=0, binding=2) readonly  buffer AuxBuffer0  { float data[]; } aux0;
  layout(set=0, binding=3) readonly  buffer AuxBuffer1  { float data[]; } aux1;
  layout(set=0, binding=4) readonly  buffer AuxBuffer2  { float data[]; } aux2;
  layout(set=0, binding=5) readonly  buffer AuxBuffer3  { float data[]; } aux3;
  layout(set=0, binding=6) readonly  buffer AuxBuffer4  { float data[]; } aux4;
  layout(set=0, binding=7) readonly  buffer AuxBuffer5  { float data[]; } aux5;

  layout(push_constant) uniform PushConstants {
      // Specific to each shader, max 128 bytes
  } pc;

=============================================================================
shader/rmsnorm.spv
=============================================================================

Purpose: RMSNorm (used by Qwen, Llama)
Math: y_i = x_i * gamma_i / RMS(x) where RMS = sqrt(mean(x^2) + eps)

Input layout:
  in0: x [batch * seq, hidden_size] (flattened, row-major)
  aux0: weight gamma [hidden_size]

Output:
  out0: y [batch * seq, hidden_size]

Push constants:
  0: uint  batch_seq     (total number of sequence positions)
  4: uint  hidden_size   (dimension size)
  8: float eps           (epsilon for numerical stability)

Dispatch:
  Local size: 256 x 1 x 1
  Num groups: (batch_seq * hidden_size + 255) / 256
  Algorithm (two-pass in one dispatch):
    Pass 1: Each thread computes x*x and atomically accumulates sum
            (use shared memory reduction: 256 -> 64 -> 16 -> 4 -> 1)
            Then compute rms = sqrt(sum / hidden_size + eps)
    Pass 2: Each thread computes y_i = x_i * gamma_i / rms
            (After barrier)

Alternate single-pass approach:
  1. Each workgroup handles ONE sequence position (seq_len workgroups)
  2. Load all hidden_size elements into shared memory (256 elements at a time)
  3. Parallel reduction for sum(x^2)
  4. Sync, compute rms
  5. Sync, compute and write y_i = x_i * gamma_i / rms
  Result: one workgroup per sequence position, perfect for transformer layers

=============================================================================
shader/rope.spv
=============================================================================

Purpose: Rotary Position Embedding
Rotates query and key vectors using complex number multiplication.

Input layout:
  in0:  query [batch, num_heads, seq, head_dim]
  aux0: key   [batch, num_kv_heads, seq, head_dim]
  aux1: cos_table [max_seq, head_dim]
  aux2: sin_table [max_seq, head_dim]

Output:
  out0: rotated query (in-place or new buffer)
  aux3: rotated key (in-place or new buffer)

Push constants:
  0:  int32_t batch
  4:  int32_t num_heads
  8:  int32_t num_kv_heads
  12: int32_t seq_len
  16: int32_t head_dim
  20: int32_t max_seq       (size of cos/sin tables)

Dispatch:
  Local size: 128 x 1 x 1
  Num groups: batch * max(num_heads, num_kv_heads) * seq_len * (head_dim / 2)
  Each workitem:
    head_idx = get_head_index()
    pos = get_position_in_seq()
    dim_pair = get_local_id(0)  // 0, 1, 2, ..., head_dim/2-1
    dim0 = dim_pair * 2
    dim1 = dim_pair * 2 + 1

    // Load rotation factors
    cos = aux1[pos * head_dim + dim_pair]
    sin = aux2[pos * head_dim + dim_pair]

    // Load values
    q0 = in0[...pos, dim0]
    q1 = in0[...pos, dim1]
    k0 = aux0[...pos, dim0]
    k1 = aux0[...pos, dim1]

    // Rotate
    q0_rot = q0 * cos - q1 * sin
    q1_rot = q0 * sin + q1 * cos
    k0_rot = k0 * cos - k1 * sin
    k1_rot = k0 * sin + k1 * cos

    // Write back
    out0 = q0_rot, q1_rot
    aux3 = k0_rot, k1_rot

=============================================================================
shader/attention_flash.spv
=============================================================================

Purpose: Flash Attention - the MAIN attention kernel
The most complex shader. Implements the online softmax algorithm.

Input layout (all in BHSD format [batch, heads, seq, head_dim]):
  in0:  query   [batch, num_heads, seq, head_dim]
  aux0: key     [batch, num_kv_heads, seq, head_dim]
  aux1: value   [batch, num_kv_heads, seq, head_dim]
  aux2: key_cache   [batch, num_kv_heads, max_seq, head_dim]
  aux3: value_cache [batch, num_kv_heads, max_seq, head_dim]

Output:
  out0: attention output [batch, num_heads, seq, head_dim]

Push constants:
  0:  int32_t batch
  4:  int32_t num_heads
  8:  int32_t num_kv_heads
  12: int32_t seq_len         (query sequence length)
  16: int32_t kv_cache_len     (length of KV cache to read from)
  20: int32_t head_dim
  24: int32_t BLOCK_M          (tile size for M dimension, e.g., 16 or 32)
  28: int32_t BLOCK_N          (tile size for N dimension, e.g., 32 or 64)
  32: int32_t is_prefill       (1 = full sequence, 0 = single token decode)
  36: int32_t num_blocks_M
  40: int32_t num_blocks_N
  44: float scale              (1.0 / sqrt(head_dim))

Dispatch:
  Local size: BLOCK_M x num_kv_heads x 1  (num_kv_heads handles GQA)
  Grid size: batch * num_query_blocks_M * 1

Algorithm (Flash Attention v2 style):

  For PREFILL (seq_len > 1):
    For each block of Q (size BLOCK_M):
      1. Load Q block into shared memory
      2. For each block of K, V (size BLOCK_N):
           - Load K, V blocks
           - Compute S_block = Q_block @ K_block^T * scale
           - Update online softmax: m_new = max(m_old, rowmax(S))
           - Update l_new = l_old * exp(m_old - m_new) + rowexp(S)
           - Update accumulator: P_block = exp(S - m_new) / l_new
           - acc += P_block @ V_block
      3. Write output block

  For DECODE (seq_len == 1, S = 1):
    One block processes the single query token:
      1. Load Q (1 token, all heads) into registers
      2. For each block of cached K, V:
           - Load K, V from KV cache
           - Compute s = Q @ K^T * scale
           - Online softmax update
           - Accumulate: acc += exp(s - m) / l * V
      3. Write output

Key optimization:
  - Use VK_KHR_cooperative_matrix if available for matrix multiply
  - Use fp16 arithmetic throughout for speed
  - Shared memory tiling to reduce global memory accesses

=============================================================================
shader/mlp_gate_up.spv
=============================================================================

Purpose: Fused gate projection + up projection + SiLU activation + multiply
Math: out = down_proj(gate(x) * silu(up(x)))

Input layout:
  in0:  input [batch * seq, hidden_size]
  aux0: gate_weight [intermediate_size, hidden_size]
  aux1: up_weight [intermediate_size, hidden_size]

Output:
  out0: gate_silu_up [batch * seq, intermediate_size]
  (The down projection is a separate shader pass)

Push constants:
  0:  uint  batch_seq      (B * S flattened)
  4:  uint  hidden_size
  8:  uint  intermediate_size
  12: uint  wg_size        (workgroup size, 256)

Dispatch:
  Local size: 256 x 1 x 1
  Num groups: batch_seq * ((intermediate_size + 255) / 256)
  Algorithm:
    Each workgroup computes one (batch, seq) position's full gate+up+SiLU:
      1. Load all hidden_size input elements (in chunks of 256)
      2. Compute partial sums:
           partial_gate = sum_i(input[i] * gate_weight[j,i]) for all i
           partial_up = sum_i(input[i] * up_weight[j,i]) for all i
      3. Reduce partial sums to final gate[j], up[j]
      4. Compute: result[j] = silu(gate[j]) * up[j]
      5. Write result[j] to output

  This is essentially a GEMM fused with SiLU activation.

=============================================================================
shader/mlp_down.spv
=============================================================================

Purpose: Down projection of the MLP
Math: out = (gate_silu_up) @ down_proj^T

Input layout:
  in0:  gate_silu_up [batch * seq, intermediate_size]
  aux0: down_weight [hidden_size, intermediate_size]

Output:
  out0: output [batch * seq, hidden_size]

Dispatch:
  Similar to mlp_gate_up but with dimensions swapped
  Uses existing gemm.spv shader internally (or can call compute_gemm directly)

=============================================================================
shader/embed.spv
=============================================================================

Purpose: Token embedding lookup (gather operation)
Input:
  in0:  embeddings [vocab_size, hidden_size]
  aux0: token_ids [seq_len] (INT32)

Output:
  out0: embedded [seq_len, hidden_size]

Push constants:
  0: uint vocab_size
  4: uint seq_len
  8: uint hidden_size

Dispatch:
  Local size: 256 x 1 x 1
  Num groups: seq_len * ((hidden_size + 255) / 256)
  Each workgroup computes one token's embedding:
    token_id = token_ids[token_idx]
    For each chunk of hidden_size (256 elements):
      Read embeddings[token_id * hidden_size + offset : offset + 256]
      Write to output

=============================================================================
shader/quantize.spv
=============================================================================

Purpose: Quantize FP16 tensor to INT8
Input:
  in0:  fp16 tensor [N]
  aux0: scales [N / block_size] (one scale per block of elements)

Output:
  out0: int8 quantized [N]

Push constants:
  0: uint N             (total elements)
  4: uint block_size    (elements per quantization block, e.g., 32 or 64)
  8: uint num_blocks
  12: int dtype_out     (8=int8, 4=int4)

Dispatch:
  Local size: 256 x 1 x 1
  Num groups: (N + 255) / 256
  Each workgroup:
    1. Find max absolute value in its block
    2. Compute scale = max_abs / 127.0
    3. Store scale to aux0[block_idx]
    4. Compute quantized = round(input / scale)
    5. Write to output
