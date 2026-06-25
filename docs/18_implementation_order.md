# Implementation Order and Integration Guide

=============================================================================
This file provides the recommended IMPLEMENTATION ORDER.
Follow this sequence to build the LLM engine incrementally, with each step
producing a working system you can test.
=============================================================================

## PHASE 1: Foundation (Week 1-2)

### Step 1.1: Extend buffer.h / buffer.c
=====================================================================
FILE: include/buffer.h, src/buffer.c (existing - extend)

Add to include/buffer.h:
  - BufferType enum: BUFFER_WEIGHT, BUFFER_ACTIVATION, BUFFER_KV_CACHE
  - buffer_alloc_type() function: allocate buffer with specific memory type hint
  - buffer_reallocate() function: resize existing buffer
  - buffer_fill_zero() function: zero out a buffer on GPU (vkCmdFillBuffer)

Add to src/buffer.c:
  - Implement the above

WHY: Your existing buffer.c is already good. This just adds convenience functions.
TEST: Load a large buffer (1GB), fill it, read it back, verify contents.

### Step 1.2: Create tensor.h / tensor.c
=====================================================================
FILES: include/tensor.h, src/tensor.c (NEW)

Purpose: Zero-copy tensor descriptor layer over Buffer.
See docs/02_tensor_h.md for full API.

WHY: Tensor descriptors let you describe multi-dimensional views over raw
     buffers without copying data. Essential for managing activation buffers.
TEST: Create tensor views of a large buffer, reshape them, verify offsets.

### Step 1.3: Create operator.h / operator.c
=====================================================================
FILES: include/operator.h, src/operator.c (NEW)

Purpose: GPU kernel registry. Centralizes shader compilation and dispatch.
See docs/03_operator_h.md for full API.

WHY: All GPU operations go through one place. Makes it easy to swap
     implementations (e.g., naive -> flash attention) without changing inference code.
TEST: Load one shader, dispatch it, verify output. Add one operator at a time.

### Step 1.4: Create the RMSNorm shader
=====================================================================
FILES: shader/rmsnorm.comp (NEW GLSL), src/operators/rmsnorm.c (NEW)

Purpose: GPU-accelerated RMSNorm. Qwen's core normalization.
See docs/17_shaders.md for shader specification.

Implementation:
  1. Write shader/rmsnorm.comp in GLSL
  2. Compile: glslangValidator -V shader/rmsnorm.comp -o bin/rmsnorm.spv
  3. Add to Makefile: bin/rmsnorm.spv
  4. Implement rmsnorm_dispatch() in src/operators/rmsnorm.c
  5. Register in operator.c

TEST: Generate random input, compute RMSNorm on CPU and GPU, compare results.
      Test shapes: [1, 1024, 3584], [1, 512, 3584], [2, 256, 3584].

---

## PHASE 2: Core Transformer (Week 2-4)

### Step 2.1: Create the RoPE shader
=====================================================================
FILES: shader/rope.comp, src/operators/rope.c

Purpose: Rotary Position Embedding. Required for Qwen's position encoding.
See docs/17_shaders.md for specification.

Implementation:
  1. First precompute cos/sin tables on CPU (in model.c)
  2. Upload tables to VRAM
  3. Write rope.comp: rotates pairs of dimensions using cos/sin
  4. Test: verify RoPE matches HuggingFace implementation

TEST: Take Qwen's Q/K tensors from Python, apply RoPE on CPU, compare with GPU output.
      Tolerance: 1e-3 for FP16.

### Step 2.2: Create the attention shader (simple first)
=====================================================================
FILES: shader/attention.comp, src/operators/attention.c

Purpose: Basic scaled dot-product attention. O(n^2) memory, but correct.
See docs/17_shaders.md for specification.

Implementation:
  1. Write attention.comp implementing: softmax(Q @ K^T / sqrt(d)) @ V
  2. Use BHSD layout [batch, heads, seq, dim]
  3. Simple three-pass approach:
       Pass 1: Q @ K^T -> attention_scores
       Pass 2: softmax(attention_scores)
       Pass 3: softmax_scores @ V -> output

TEST: Small attention: batch=1, heads=4, seq=128, dim=32.
      Compare with PyTorch: torch.nn.functional.scaled_dot_product_attention

### Step 2.3: Create the Flash Attention shader
=====================================================================
FILES: shader/attention_flash.comp, src/operators/attention_flash.c

Purpose: Memory-efficient attention. Replaces naive attention for production.
See docs/17_shaders.md for specification.

Implementation:
  1. Implement Flash Attention v2 algorithm
  2. Support both prefill (full sequence) and decode (single token) modes
  3. Handle GQA (num_kv_heads < num_q_heads) by broadcasting K/V across groups
  4. Support sliding window (optional for Qwen3.5)

TEST: Flash attention output must match naive attention exactly (within FP16 tolerance).
      Benchmark: flash should be faster at seq_len > 256.

### Step 2.4: Create the MLP shader
=====================================================================
FILES: shader/mlp_gate_up.comp, shader/mlp_down.comp,
       src/operators/mlp.c

Purpose: Feed-forward network with fused SiLU activation.
See docs/17_shaders.md for specification.

Implementation:
  1. mlp_gate_up.comp: gate_proj + up_proj + silu(gate) * up
  2. mlp_down.comp: down_proj (reuse existing GEMM or write new shader)
  3. Both pass through operator.c

TEST: Generate random input, run MLP on CPU and GPU, compare.
      Test: [1, 512, 3584] input -> [1, 512, 18944] intermediate -> [1, 512, 3584] output

### Step 2.5: Create model.c
=====================================================================
FILE: src/model.c

Purpose: Load Qwen3.5-9B weights from safetensors into VRAM.
See docs/10_model_c.md for full specification.

Implementation:
  1. Parse config.json from model directory
  2. Use safetensors Python or a C safetensors parser (miniz for zip + custom tensor parse)
     NOTE: For Phase 1, the easiest approach is to call Python:
           python -c "import safetensors; safetensors.torch.save_file(...)"
           to convert to raw binary, then load in C.
           For proper Phase 2, parse safetensors format directly in C.
  3. Allocate VRAM buffers for all weights
  4. Upload layer by layer (stream to VRAM via staging buffers)
  5. Pre-allocate KV cache buffers
  6. Precompute RoPE tables

TEST: Load model, print info (layer count, param count, memory usage).
      Verify weights loaded correctly by reading back a few values and comparing.

---

## PHASE 3: Full Forward Pass (Week 4-6)

### Step 3.1: Create inference.c - embedding lookup
=====================================================================
FILE: src/inference.c

Start with the embedding lookup:
  - Load embed_tokens buffer
  - Write embed.spv shader (gather operation)
  - Test: encode "hello" in Python with HF tokenizer, run embed shader,
          verify output matches PyTorch embedding lookup.

### Step 3.2: Create inference.c - single layer forward
=====================================================================
Implement layer_forward() for ONE layer:
  1. RMSNorm (input layernorm)
  2. QKV GEMM (reuse existing compute.c GEMM)
  3. Split Q, K, V
  4. RoPE on Q and K
  5. Flash Attention
  6. O projection + residual add
  7. RMSNorm (post-attention layernorm)
  8. MLP (gate_up + down)
  9. Residual add

TEST: Run single layer on random input, compare with PyTorch layer-by-layer.
      Debug: if outputs don't match, check each step in isolation.

### Step 3.3: Create inference.c - full model forward
=====================================================================
Extend inference_forward() to loop through all layers:
  1. Embedding lookup
  2. For each layer: layer_forward()
  3. Final RMSNorm
  4. LM head (logits)
  5. Read back logits for the last position

TEST: Run full model on a known prompt.
      Compare logits with HuggingFace PyTorch model.
      Check top-1 token matches.

### Step 3.4: Create tokenizer.c
=====================================================================
FILE: src/tokenizer.c

Purpose: Convert text to token IDs and back.
See docs/15_tokenizer_c.md for implementation approaches.

Recommendation for Phase 1: Use subprocess approach.
  1. Write tokenizer.py that wraps HuggingFace tokenizer
  2. C code calls: python tokenizer.py encode "text"
  3. Parse output token IDs

TEST: encode("Hello, world!") -> compare with HF tokenizer.
      decode(token_ids) -> compare with HF tokenizer.

### Step 3.5: Autoregressive generation
=====================================================================
Implement inference_generate():
  1. Prefill: forward pass on full prompt
  2. Sample first token from logits
  3. Decode loop:
       - Forward on single new token
       - Sample next token
       - Stop at EOS or max_tokens

TEST: Generate from "Once upon a time".
      Compare first 10 tokens with HuggingFace generate().

---

## PHASE 4: Polish and Optimization (Week 6-8+)

### Step 4.1: Create engine.c
=====================================================================
FILE: src/engine.c

Wrap everything in the LLMEngine facade.
See docs/14_engine_c.md.

TEST: llm_engine_create() -> llm_engine_generate() -> llm_engine_destroy()
      Should produce coherent text from a prompt.

### Step 4.2: Add sampling options
=====================================================================
  - Greedy (argmax)
  - Temperature sampling
  - Top-K filtering
  - Top-P (nucleus) sampling
  - Repetition penalty

### Step 4.3: Performance optimization
=====================================================================
  - Operator fusion (fuse rmsnorm + add into one shader)
  - Memory pooling for activations (reuse buffers instead of allocating)
  - Batch multiple sequences together (if batch size > 1)
  - KV cache page management

### Step 4.4: Quantization (if needed)
=====================================================================
  - Add quantize.spv and dequantize.spv shaders
  - Calibrate quantization scales on representative dataset
  - Test INT8 and INT4 inference

---

## Build System (Makefile additions)

Add to Makefile:

  # New source files
  NEW_SRC = src/model.c src/tensor.c src/operator.c src/inference.c \\
            src/tokenizer.c src/engine.c \\
            src/operators/layernorm.c src/operators/rmsnorm.c \\
            src/operators/rope.c src/operators/attention.c \\
            src/operators/attention_flash.c src/operators/mlp.c \\
            src/operators/softmax.c src/operators/quantize.c

  # New shader targets
  NEW_SPIRV = bin/rmsnorm.spv bin/rope.spv bin/attention.spv \\
              bin/attention_flash.spv bin/mlp_gate_up.spv \\
              bin/mlp_down.spv bin/embed.spv bin/quantize.spv \\
              bin/layernorm.spv bin/softmax.spv

  # Build rules
  bin/rmsnorm.spv: shader/rmsnorm.comp | bin
      $(GLSLC) $(GLSLCFLAGS) -o $@ $<
  ... (one rule per shader)

  main: $(NEW_SRC) $(NEW_SPIRV) $(EXISTING_DEPS)
      $(CC) $(CFLAGS) $(NEW_SRC) $(EXISTING_SRC) $(LDFLAGS) -o bin/main.exe

---

## Quick-Start Checklist

  [ ] Step 1.1: Extend buffer.c
  [ ] Step 1.2: Create tensor.c
  [ ] Step 1.3: Create operator.c
  [ ] Step 1.4: RMSNorm shader
  [ ] Step 2.1: RoPE shader
  [ ] Step 2.2: Attention shader (naive)
  [ ] Step 2.3: Flash Attention shader
  [ ] Step 2.4: MLP shader
  [ ] Step 2.5: model.c (loading)
  [ ] Step 3.1: Embedding lookup
  [ ] Step 3.2: Single layer forward
  [ ] Step 3.3: Full model forward
  [ ] Step 3.4: Tokenizer
  [ ] Step 3.5: Generation loop
  [ ] Step 4.1: engine.c facade
  [ ] Step 4.2: Sampling options
  [ ] Step 4.3: Optimization
  [ ] Step 4.4: Quantization
