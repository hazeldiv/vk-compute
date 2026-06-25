# src/inference.c - Forward Pass and Generation Loop

=============================================================================
FILE: src/inference.c
PURPOSE: Orchestrates the transformer forward pass. Manages KV cache.
         Implements autoregressive generation with sampling.
=============================================================================

## Dependencies
- include/inference.h
- include/model.h
- include/operator.h
- include/tensor.h
- include/compute.h (for GEMM)

## Key Functions

-------------------------------------------------------------------------------
static int build_scratch_tensors(InferenceContext* ctx)
-------------------------------------------------------------------------------
  - Pre-allocates named views into the scratch buffer
  - Scratch is one large VRAM buffer big enough for the largest intermediate
  - Named scratch tensors:
      SCRATCH_HIDDEN:   [max_batch, max_seq, hidden_size]  hidden states
      SCRATCH_QKV:     [max_batch, max_seq, (q+kv+kv) * head_dim] QKV concat
      SCRATCH_Q:       [max_batch, num_q_heads, max_seq, head_dim]
      SCRATCH_K:       [max_batch, num_kv_heads, max_seq, head_dim]
      SCRATCH_V:       [max_batch, num_kv_heads, max_seq, head_dim]
      SCRATCH_SCORES:  [max_batch, num_heads, max_seq, max_seq]  attn scores
      SCRATCH_ATTN_OUT:[max_batch, num_q_heads, max_seq, head_dim]
      SCRATCH_MLP_IN:  [max_batch, max_seq, intermediate_size]
      SCRATCH_MLP_OUT: [max_batch, max_seq, hidden_size]
      SCRATCH_LOGITS:  [max_batch, max_seq, vocab_size]

  - All scratch tensors share the same physical buffer (offset views)
  - Benefit: zero allocation during inference, single vkCmdFillBuffer reset

-------------------------------------------------------------------------------
InferenceContext* inference_context_create(const char* model_path,
                                            Device* device,
                                            CommandPool* cmd_pool,
                                            int32_t max_seq_len)
-------------------------------------------------------------------------------
  PUBLIC API
  1. Allocate InferenceContext*
  2. model_load_from_directory() -> ctx->model
  3. op_registry_create(device, "shader/") -> ctx->registry
  4. ctx->cmd_pool = cmd_pool
  5. Pre-allocate descriptor pools for all operator types
  6. allocate_kv_cache() -> ctx->kv_cache
  7. build_scratch_tensors()
  8. precompute_rope_tables() -> ctx->cos_table, ctx->sin_table
  9. ctx->kv_cache_current_len = 0
  10. Return ctx

-------------------------------------------------------------------------------
void inference_context_destroy(InferenceContext* ctx)
-------------------------------------------------------------------------------
  1. op_registry_destroy(ctx->registry)
  2. model_unload(ctx->model)
  3. Destroy scratch buffer
  4. Destroy RoPE tables
  5. Destroy descriptor pools
  6. Free(ctx)

-------------------------------------------------------------------------------
static void layer_forward(InferenceContext* ctx, CommandBuffer* cmd,
                          int32_t layer_idx,
                          Tensor* x,           // [batch, seq, hidden] in/out
                          Tensor* output_logits) // NULL unless last layer
-------------------------------------------------------------------------------
  INTERNAL - forward through one transformer layer
  Called from inference_forward() for each layer.

  ModelConfig* cfg = &ctx->model->config;
  LayerWeights* layer = &ctx->model->layers[layer_idx];
  int32_t B = 1;  // batch size
  int32_t S = x->shape[1];  // current sequence length
  int32_t H = cfg->hidden_size;
  int32_t QH = cfg->num_attention_heads;
  int32_t KH = cfg->num_key_value_heads;
  int32_t D = cfg->head_dim;
  int32_t IS = cfg->intermediate_size;

  Step 1: input_layernorm
    rmsnorm(input=x, weight=layer->input_layernorm, eps=cfg->norm_eps)
    -> SCRATCH_NORMED

  Step 2: QKV projection (GEMM)
    gemm(SCRATCH_NORMED, layer->q_proj) -> SCRATCH_QKV  [B, S, (QH+KH+KH)*D]
    Note: q_proj, k_proj, v_proj may be concatenated or separate.
    If concatenated: split into SCRATCH_Q, SCRATCH_K, SCRATCH_V

  Step 3: Split Q, K, V from concatenated QKV buffer
    (memcpy via staging or buffer view)

  Step 4: Apply RoPE to Q and K
    op_rope(SCRATCH_Q, SCRATCH_K) -> SCRATCH_Q (in-place), SCRATCH_K (in-place)
    Uses precomputed cos/sin tables from ctx

  Step 5: Update KV cache
    if (is_prefill):
      Copy SCRATCH_K into ctx->kv_cache.key_cache[layer_idx] at offset S
      Copy SCRATCH_V into ctx->kv_cache.value_cache[layer_idx] at offset S
    else:
      Copy SCRATCH_K into ctx->kv_cache.key_cache[layer_idx] at offset current_len
      Copy SCRATCH_V into ctx->kv_cache.value_cache[layer_idx] at offset current_len

  Step 6: Flash Attention
    op_attention_flash(
        query=SCRATCH_Q,
        key_cache=ctx->kv_cache.key_cache[layer_idx],
        value_cache=ctx->kv_cache.value_cache[layer_idx],
        output=SCRATCH_ATTN_OUT
    )
    Note: for decode (S=1), this just computes attention on the single token
    using the full KV cache. For prefill, computes attention over full sequence.

  Step 7: Output projection + residual add
    gemm(SCRATCH_ATTN_OUT, layer->o_proj) -> SCRATCH_ATTN_PROJ
    op_add_residual(SCRATCH_ATTN_PROJ, x) -> SCRATCH_RESIDUAL1
    (x now holds the attention residual)

  Step 8: post_attention_layernorm
    rmsnorm(SCRATCH_RESIDUAL1, layer->post_attention_layernorm)
    -> SCRATCH_NORMED2

  Step 9: MLP
    op_mlp(
        input=SCRATCH_NORMED2,
        gate=layer->gate_proj,
        up=layer->up_proj,
        down=layer->down_proj,
        output=SCRATCH_MLP_OUT
    )
    Gate and up are fused in the MLP shader:
      tmp = silu(gate_proj * input)
      SCRATCH_MLP_OUT = down_proj * tmp

  Step 10: Residual add
    op_add_residual(SCRATCH_MLP_OUT, SCRATCH_RESIDUAL1)
    -> x (overwrites input buffer, ready for next layer)

-------------------------------------------------------------------------------
int inference_forward(InferenceContext* ctx,
                      const int32_t* token_ids,
                      int32_t seq_len,
                      float* output_logits)
-------------------------------------------------------------------------------
  PUBLIC API - runs one full forward pass
  1. Reset scratch buffer (vkCmdFillBuffer with 0)
  2. ctx->kv_cache_current_len = seq_len  (for prefill)

  // === Embedding lookup ===
  embedded = SCRATCH_HIDDEN
  For each position i in [0, seq_len):
    Gather embedding from ctx->model->embeddings.embed_tokens
    at index token_ids[i]
  This is a gather operation: output[i] = embeddings[token_ids[i]]
  Implemented as a shader or as a series of small copies.

  // === Pass through all layers ===
  x = SCRATCH_HIDDEN  // [1, seq_len, hidden_size]
  for layer_idx = 0 to ctx->model->config.num_hidden_layers - 1:
    layer_forward(ctx, cmd, layer_idx, &x, NULL)

  // === Final layer norm ===
  rmsnorm(x, final_layernorm, eps) -> SCRATCH_NORMED

  // === LM Head (logits) ===
  gemm(SCRATCH_NORMED, ctx->model->embeddings.lm_head) -> SCRATCH_LOGITS
  If sharing embeddings: use embed_tokens instead of lm_head

  // === Optional: read back logits ===
  if (output_logits != NULL):
    readBuffer(SCRATCH_LOGITS, output_logits, vocab_size * sizeof(float))
    (Read only the last position's logits for generation)

  3. Return 0

-------------------------------------------------------------------------------
static int32_t sample_token(const float* logits, int32_t vocab_size,
                             SamplingParams* params)
-------------------------------------------------------------------------------
  INTERNAL - pure CPU sampling
  1. Apply temperature: logits[i] /= temperature (if temp > 0)
  2. Apply top_k: zero out logits outside top k
  3. Apply top_p: sort, accumulate until probability > top_p
  4. Convert to probabilities: softmax(logits)
  5. Sample from distribution (or argmax for greedy)
  6. Return token ID

-------------------------------------------------------------------------------
int inference_generate(InferenceContext* ctx,
                       const int32_t* prompt_tokens,
                       int32_t prompt_len,
                       SamplingParams* params,
                       GenerationResult* result)
-------------------------------------------------------------------------------
  PUBLIC API - full autoregressive generation
  1. Initialize result struct, allocate result->token_ids[]
  2. Copy prompt_tokens to result->token_ids
  3. ctx->kv_cache_current_len = 0

  // === Prefill phase ===
  inference_forward(ctx, prompt_tokens, prompt_len, NULL)
  // KV cache now contains keys/values for all prompt tokens
  // Logits at last position are in SCRATCH_LOGITS

  readBuffer(last_logits) -> float[vocab_size]
  sampled = sample_token(last_logits, vocab_size, params)
  Append sampled to result->token_ids

  // === Decode phase (token-by-token) ===
  while (result->num_tokens < params->max_new_tokens):
    // Update KV cache position for the new token
    ctx->kv_cache_current_len = prompt_len + result->num_tokens

    // Forward pass on single new token
    inference_forward(ctx, &sampled, 1, logits)
    // (KV cache grows by 1 each iteration)

    sampled = sample_token(logits, vocab_size, params)
    if (sampled == params->eos_token_id && result->num_tokens >= params->min_new_tokens)
        break
    Append sampled to result->token_ids

  4. Return 0

-------------------------------------------------------------------------------
void inference_reset_kv_cache(InferenceContext* ctx)
-------------------------------------------------------------------------------
  - ctx->kv_cache_current_len = 0
  - No need to clear actual memory; position tracking prevents using stale data

-------------------------------------------------------------------------------
int inference_embed(InferenceContext* ctx, CommandBuffer* cmd,
                    const int32_t* token_ids, int32_t seq_len, Tensor* output)
-------------------------------------------------------------------------------
  PUBLIC API - standalone embedding lookup
  - Gather from embed_tokens buffer at given indices
  - Implemented as a GPU shader that reads from the embedding buffer
  - Alternative: pre-copy embeddings to staging and do table lookup on CPU

## KV Cache Update Strategy

  For PREFILL (seq_len > 1):
    - Compute Q, K, V for all tokens in the sequence
    - After RoPE, write K to key_cache[layer][:, :, :seq_len, :]
    - Write V to value_cache[layer][:, :, :seq_len, :]
    - Run Flash Attention on the full K/V tensors

  For DECODE (seq_len == 1):
    - Compute Q for the single new token
    - Read K, V for the new token
    - Write to key_cache[layer][:, :, current_pos, :]  (one row)
    - Write to value_cache[layer][:, :, current_pos, :] (one row)
    - Run Flash Attention: Q vs full cached K/V
    - No need to recompute attention for previous tokens
