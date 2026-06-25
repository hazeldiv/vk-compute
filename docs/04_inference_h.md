# include/inference.h - Forward Pass and Generation

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "model.h"
#include "operator.h"
#include "tensor.h"

// ---------------------------------------------------------------------------
// InferenceContext
// ---------------------------------------------------------------------------
// Holds everything needed for inference:
// - Model weights (loaded in VRAM)
// - KV cache buffers
// - Scratch/activation buffers
// - Operator registry
// - Command pool for GPU work
//
// ONE InferenceContext per loaded model.
// Not thread-safe (use one per thread, or add mutex).
// ---------------------------------------------------------------------------

typedef struct {
    // Model
    ModelState* model;

    // Operator registry (all GPU kernels)
    OpRegistry* registry;

    // Command pool (from existing engine)
    CommandPool* cmd_pool;

    // Descriptor set pool (pre-allocated sets for reuse)
    DescriptorPool* desc_pool;

    // KV Cache management
    int32_t kv_cache_max_seq_len;
    int32_t kv_cache_current_len;  // Number of cached tokens

    // Scratch buffers for intermediate tensors
    // Pre-allocated once, reused every forward pass.
    // Size determined by model config (largest intermediate tensor).
    Buffer* scratch;               // Large scratch buffer
    Tensor* scratch_tensors;       // Named views into scratch

    // Pre-allocated descriptor sets (one per operator type)
    // Reused across all forward passes to avoid allocation overhead.
    DescriptorSet* descriptor_sets[OP_ROTARY_EMBED + 1];

    // Precomputed position tables
    Buffer cos_table;   // [max_seq, head_dim] for RoPE
    Buffer sin_table;   // [max_seq, head_dim] for RoPE

    // Current sequence state
    int32_t current_seq_len;
    int32_t max_batch_size;
} InferenceContext;

// ---------------------------------------------------------------------------
// Generation result
// ---------------------------------------------------------------------------

typedef struct {
    int32_t* token_ids;   // Output token IDs [num_tokens]
    int32_t num_tokens;   // How many tokens were generated
    float* logits;        // Optional: final logits for analysis
    double time_total_ms; // Wall-clock time for generation
    double time_per_token_ms; // Average time per generated token
} GenerationResult;

// ---------------------------------------------------------------------------
// Sampling parameters
// ---------------------------------------------------------------------------

typedef struct {
    // Sampling method
    int type;             // 0=greedy, 1=temperature, 2=top_k, 3=top_p
    float temperature;    // For temperature sampling (0.0 = greedy)
    int top_k;            // Top-K filtering (0 = disabled)
    float top_p;          // Top-P (nucleus) sampling (0.0-1.0)

    // Repetition penalty
    bool use_repetition_penalty;
    float repetition_penalty;
    int repeat_penalty_window;  // How many past tokens to check

    // Generation limits
    int32_t max_new_tokens;
    int32_t min_new_tokens;
    int32_t max_total_tokens;  // Including prompt

    // EOS handling
    int32_t eos_token_id;
    bool ignore_eos;           // Force max_new_tokens regardless of EOS
} SamplingParams;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * inference_context_create()
 * Creates an inference context for the given model.
 * Uploads weights to VRAM, pre-allocates KV cache and scratch buffers,
 * precomputes RoPE tables, and compiles operator shaders.
 *
 * @param model_path   Path to model directory (same as model_load_from_directory)
 * @param device       Vulkan device
 * @param cmd_pool     Command pool
 * @param max_seq_len  Maximum sequence length (determines KV cache size)
 * @return             InferenceContext* or NULL on failure
 */
InferenceContext* inference_context_create(
    const char* model_path,
    Device* device,
    CommandPool* cmd_pool,
    int32_t max_seq_len
);

/**
 * inference_context_destroy()
 * Frees all resources.
 */
void inference_context_destroy(InferenceContext* ctx);

/**
 * inference_forward()
 * Runs one full forward pass of the model on a token sequence.
 * Used for both prefill (input tokens -> hidden states) and
 * decode (single new token -> next token logits).
 *
 * @param ctx          Inference context
 * @param token_ids    Input token IDs [seq_len]
 * @param seq_len      Length of token_ids
 * @param output_logits  Output logits [vocab_size] (can be NULL to discard)
 * @return             0 on success
 *
 * Forward pass sequence per layer:
 *   1. embedded = embed_tokens[token_ids]           // Gather from embedding buffer
 *   2. x = embedded
 *   3. for layer = 0 .. num_layers-1:
 *        residual = x
 *        x_norm = rmsnorm(x, layer.input_layernorm)
 *        qkv = x_norm @ layer.qkv_proj               // GEMM
 *        q, k, v = split(qkv)
 *        q = rope(q)
 *        k = rope(k)
 *        attn_out = attention(q, k, v, kv_cache[layer])
 *        x = residual + (attn_out @ layer.o_proj)     // GEMM + add
 *        residual = x
 *        x_norm = rmsnorm(x, layer.post_attn_layernorm)
 *        mlp_out = mlp(x_norm, layer.gate, layer.up, layer.down)
 *        x = residual + mlp_out                       // add
 *   4. final_norm = rmsnorm(x, final_layernorm)
 *   5. logits = final_norm @ lm_head                 // GEMM
 */
int inference_forward(
    InferenceContext* ctx,
    const int32_t* token_ids,
    int32_t seq_len,
    float* output_logits   // Optional, can be NULL
);

/**
 * inference_generate()
 * Autoregressive generation loop.
 * Calls inference_forward() repeatedly, appending one token at a time.
 * Manages KV cache automatically.
 *
 * @param ctx           Inference context
 * @param prompt_tokens Input token IDs [prompt_len]
 * @param prompt_len    Length of prompt
 * @param params        Sampling parameters
 * @param result        Output: generated tokens (caller must free token_ids)
 * @return              0 on success
 */
int inference_generate(
    InferenceContext* ctx,
    const int32_t* prompt_tokens,
    int32_t prompt_len,
    SamplingParams* params,
    GenerationResult* result
);

/**
 * inference_reset_kv_cache()
 * Clears the KV cache for a fresh generation.
 * Call this before starting a new conversation / generation.
 */
void inference_reset_kv_cache(InferenceContext* ctx);

/**
 * inference_embed()
 * Converts token IDs to embedding vectors.
 * Standalone function for embedding-only use cases.
 *
 * @param ctx        Inference context
 * @param cmd        Command buffer
 * @param token_ids  Token IDs [seq_len]
 * @param seq_len    Sequence length
 * @param output     Output tensor [seq_len, hidden_size]
 */
int inference_embed(
    InferenceContext* ctx,
    CommandBuffer* cmd,
    const int32_t* token_ids,
    int32_t seq_len,
    Tensor* output
);

/**
 * inference_sample_token()
 * Samples the next token from logits using the given parameters.
 * Pure CPU function, no GPU involvement.
 *
 * @param logits    Logits array [vocab_size]
 * @param params    Sampling parameters
 * @return           Sampled token ID
 */
int32_t inference_sample_token(const float* logits, SamplingParams* params);
