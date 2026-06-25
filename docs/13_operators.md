# src/operators/ - Individual GPU Kernel Implementations

=============================================================================
This directory contains one .c file per GPU compute shader.
Each file implements the shader compilation and the CPU-side dispatch wrapper.
=============================================================================

## src/operators/layernorm.c
Purpose: Standard LayerNorm
Math: y = (x - mean) / sqrt(variance + eps) * gamma + beta

Functions:
  - layernorm_compile_shader(): loads layernorm.spv, creates pipeline
  - layernorm_dispatch(): binds buffers, pushes {hidden_size, seq_len, eps},
    dispatches with workgroup=128

Push constants:
  uint hidden_size, uint seq_len, float eps

Dispatch:
  Global workitems = batch * seq_len
  Each workitem computes one output position: reduces over hidden_size dimension
  Two-pass approach (or tree-reduction in shared memory)


## src/operators/rmsnorm.c
Purpose: RMSNorm (no mean, only RMS)
Math: y = x / RMS(x) * gamma  where RMS = sqrt(mean(x^2) + eps)
Used by: Qwen, Llama, Mistral (faster than LayerNorm, no bias)

Functions:
  - rmsnorm_compile_shader()
  - rmsnorm_dispatch()

Push constants:
  uint hidden_size, uint seq_len, float eps

Dispatch:
  Global = batch * seq_len
  One pass: each workitem computes RMS for its sequence position,
  then scales all hidden values
  Uses shared memory reduction for the sum(x^2) step


## src/operators/rope.c
Purpose: Rotary Position Embedding (RoPE)
Applies rotation to query and key vectors in frequency domain.
Reference: https://arxiv.org/abs/2104.09864

Math:
  For dimension pair (d, d+1) at position p:
    x_d' = x_d * cos(p * theta_d) - x_{d+1} * sin(p * theta_d)
    x_{d+1}' = x_d * sin(p * theta_d) + x_{d+1} * cos(p * theta_d)
  where theta_d = base * (base_ratio ^ (d / head_dim))

Functions:
  - rope_compile_shader()
  - rope_dispatch()

Push constants:
  int32_t seq_len, int32_t head_dim, int32_t num_heads,
  int32_t num_kv_heads, double base

Bindings:
  binding 0: query input buffer
  binding 1: query output buffer
  binding 2: key input buffer
  binding 3: key output buffer
  binding 4: cos_table
  binding 5: sin_table

Dispatch:
  Global = batch * num_heads * seq_len * (head_dim / 2)
  Each workitem handles one position's one dimension pair
  Reads cos/sin from precomputed VRAM tables (cache-friendly)


## src/operators/attention.c
Purpose: Naive O(n^2) scaled dot-product attention
Math: softmax(Q @ K^T / sqrt(d)) @ V
Use: Testing only, or very short sequences. For production, use attention_flash.c

Functions:
  - attention_compile_shader()
  - attention_dispatch()

Push constants:
  int32_t batch, int32_t num_heads, int32_t seq_len, int32_t head_dim

Dispatch:
  3D grid: [batch, num_heads, seq_len]
  Each (b, h, s_q) thread computes the full attention output for that query
  O(seq_len^2) loads of K and V - very memory intensive

WARNING: Memory usage = O(batch * num_heads * seq_len^2)
For seq_len=2048, batch=1, num_heads=32: ~512MB for attention scores alone.


## src/operators/attention_flash.c
Purpose: Flash Attention algorithm (more important than naive attention)
Implements the online softmax algorithm with tiling.
Reference: https://arxiv.org/abs/2205.14135

Key ideas:
  - Process Q, K, V in blocks (tiles)
  - Maintain running statistics (m, l) for online softmax
  - Never materializes the full O(n^2) attention matrix
  - Memory = O(n) instead of O(n^2)

Functions:
  - attention_flash_compile_shader()
  - attention_flash_dispatch()

Push constants:
  int32_t batch, int32_t num_heads, int32_t num_kv_heads,
  int32_t seq_len, int32_t head_dim, int32_t layer_idx,
  int32_t is_prefill, int32_t sliding_window

Dispatch:
  For PREFILL (seq_len > 1):
    Grid: [batch, num_heads, num_blocks_M]
    Each block computes one row-block of the attention output
    Block sizes: BLOCK_M=16, BLOCK_N=32 (tunable)
    Pass 1: forward pass, accumulate stats
    Pass 2: backward pass (optional), finalize outputs

  For DECODE (seq_len == 1):
    Grid: [batch, num_heads]
    Each block computes attention for one query token against full KV cache
    Much faster for single-token decode since we avoid recomputing old tokens


## src/operators/mlp.c
Purpose: Feed-forward network (FFN / MLP)
Architecture: gate_proj(x) -> gate; up_proj(x) -> up; silu(gate)*up -> act; down_proj(act) -> out

Fused into two shader passes:

  PASS 1: Gate + Up projection (fused)
    Input: x [B, S, hidden_size]
    Output: act [B, S, intermediate_size]
    gate = x @ gate_proj^T
    up = x @ up_proj^T
    act = silu(gate) * up
    All three operations fused in one shader to reduce memory bandwidth.

  PASS 2: Down projection
    Input: act [B, S, intermediate_size]
    Output: out [B, S, hidden_size]
    out = act @ down_proj^T

Functions:
  - mlp_compile_shader()
  - mlp_dispatch_gate_up()  - fused gate + up + silu
  - mlp_dispatch_down()     - down projection

Push constants (gate_up):
  int32_t B, int32_t S, int32_t hidden_size, int32_t intermediate_size

Dispatch gate_up:
  Global = B * S * (intermediate_size / 256)
  Workgroup = 256x1x1
  Each workgroup computes one (B,S) position's gate+up+silu

Dispatch down:
  Global = B * S * (hidden_size / 256)
  Workgroup = 256x1x1


## src/operators/softmax.c
Purpose: Softmax over specified axis
Most often: softmax over attention scores axis=-1 (last dimension)
Math: softmax(x_i) = exp(x_i) / sum(exp(x_j))

Usually fused into the attention shader (attention_flash.c).
Standalone softmax exists as a fallback or for debugging.

Functions:
  - softmax_compile_shader()
  - softmax_dispatch()

Dispatch:
  Global = batch * num_heads * seq_len * seq_len / 256
  Workgroup = 256x1x1
  Two-phase:
    Phase 1: find max (reduction in shared memory)
    Phase 2: compute exp(sum) and normalize


## src/operators/quantize.c
Purpose: Weight quantization for INT8/INT4 inference
Needed when running quantized models (Qwen3.5-9B-Q4)

Functions:
  - quantize_dispatch(): FP16 -> INT8/INT4
  - dequantize_dispatch(): INT8/INT4 -> FP16

INT4 packing:
  One byte holds 2 INT4 values: byte = (val1 << 4) | val2
  Handle endianness carefully for consistent results.

Quantization formula:
  scale = max(abs(tensor)) / 127.0  (for INT8 symmetric)
  quantized = round(tensor / scale)
  dequantized = quantized * scale

Dispatch (quantize):
  Global = tensor_size / elements_per_workgroup
  Each workgroup processes one block, finds max, computes scale, quantizes

Dispatch (dequantize):
  Global = num_packed_bytes / elements_per_workgroup
  Each workitem unpacks 2 INT4 values and multiplies by scale
