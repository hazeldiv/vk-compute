# include/model.h - Model Config, Weights, and State

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "buffer.h"

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_BFLOAT16,
    DTYPE_INT8,
    DTYPE_INT4,
} DataType;

typedef enum {
    ATTN_TYPE_GATED,       // Standard attention with gating
    ATTN_TYPE_FLASH,        // Flash Attention
    ATTN_TYPE_GROUPED,      // Grouped Query Attention (GQA)
} AttentionType;

typedef enum {
    NORM_TYPE_RMSNORM,      // RMSNorm (no mean, only variance) - used by Qwen
    NORM_TYPE_LAYERNORM,    // Standard LayerNorm (mean + variance)
} NormType;

// ---------------------------------------------------------------------------
// ModelConfig - architecture description (read from config.json)
// ---------------------------------------------------------------------------

typedef struct {
    // Architecture name
    char model_name[128];

    // Vocabulary
    int32_t vocab_size;
    int32_t hidden_size;       // Model dimension (e.g., 3584 for Qwen3.5-9B)
    int32_t intermediate_size; // FFN intermediate dimension
    int32_t num_hidden_layers;
    int32_t num_attention_heads;
    int32_t num_key_value_heads; // GQA: if < num_attention_heads, use GQA
    int32_t head_dim;           // Usually 128 or 64

    // Precision
    DataType weight_dtype;
    DataType compute_dtype;    // Actual dtype used for compute

    // Position encoding
    int32_t max_position_embeddings;
    double rope_theta;          // RoPE base frequency

    // Normalization
    NormType norm_type;
    double norm_eps;

    // Activations
    char hidden_act[32];        // "silu", "gelu", "relu"

    // Attention
    AttentionType attn_type;
    bool use_sliding_window;
    int32_t sliding_window;

    // Quantization (if applicable)
    bool quantized;
    char quant_method[16];     // "fp16", "int8", "nf4"
} ModelConfig;

// ---------------------------------------------------------------------------
// LayerWeights - all weight matrices for ONE transformer layer
// ---------------------------------------------------------------------------

typedef struct {
    // Attention weights
    Buffer q_proj;   // [hidden_size, num_q_heads * head_dim]     (or compressed)
    Buffer k_proj;   // [hidden_size, num_kv_heads * head_dim]
    Buffer v_proj;   // [hidden_size, num_kv_heads * head_dim]
    Buffer o_proj;   // [num_kv_heads * head_dim, hidden_size]

    // MLP weights
    Buffer gate_proj; // [intermediate_size, hidden_size]
    Buffer up_proj;   // [intermediate_size, hidden_size]
    Buffer down_proj; // [hidden_size, intermediate_size]

    // Layer norms (some models have these as separate weights per layer)
    Buffer input_layernorm;     // Applied before attention
    Buffer post_attention_layernorm;  // Applied before MLP

    // Optional: quantized weight metadata
    void* quant_metadata;
} LayerWeights;

// ---------------------------------------------------------------------------
// Embeddings - shared embeddings and output head
// ---------------------------------------------------------------------------

typedef struct {
    Buffer embed_tokens;        // [vocab_size, hidden_size], FP16/FP32
    Buffer lm_head;              // [vocab_size, hidden_size], often shared with embed_tokens
} Embeddings;

// ---------------------------------------------------------------------------
// KVCache - persistent cache for autoregressive decoding
// ---------------------------------------------------------------------------

typedef struct {
    Buffer* key_cache;   // [num_layers][num_kv_heads, max_seq_len, head_dim]
    Buffer* value_cache; // [num_layers][num_kv_heads, max_seq_len, head_dim]
    int32_t max_seq_len;
    int32_t current_seq_len; // How many tokens are cached
} KVCache;

// ---------------------------------------------------------------------------
// ModelState - all weights and buffers for the entire model
// ---------------------------------------------------------------------------

typedef struct {
    ModelConfig config;

    // Weights
    Embeddings embeddings;
    LayerWeights* layers;  // [num_hidden_layers]

    // Optional: shared expert routing (for MoE models)
    void* moe_state;

    // KV Cache
    KVCache kv_cache;

    // Scratch/activation buffers (reused per forward pass)
    Buffer activations;      // Pre-allocated scratch space for intermediate tensors
    Buffer hidden_states;   // [batch, seq_len, hidden_size]
    Buffer attention_scores; // [batch, num_heads, seq_len, seq_len]
    Buffer output_buffer;   // For logits

    // Descriptor set layout for operators (shared across layers)
    // Built once, reused for all dispatches
    void* descriptor_layout;

    // Physical memory info
    uint64_t total_vram_used;
    uint64_t total_vram_available;
} ModelState;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * model_load_from_directory()
 * Loads model config and weights from a directory (e.g., model/Qwen3.5-9B/).
 * Reads config.json for architecture, loads safetensors for weights.
 * Uploads all weights to VRAM (MEMORY_VRAM).
 *
 * @param dir_path     Path to model directory
 * @param device       Vulkan device from container
 * @param out_state    Output: populated ModelState
 * @return             0 on success, negative on error
 */
int model_load_from_directory(
    const char* dir_path,
    Device* device,
    ModelState** out_state
);

/**
 * model_unload()
 * Frees all VRAM buffers and host memory associated with the model.
 */
void model_unload(ModelState* state);

/**
 * model_get_config()
 * Returns the model configuration.
 */
ModelConfig* model_get_config(ModelState* state);

/**
 * model_print_info()
 * Prints model size, memory usage, layer breakdown.
 */
void model_print_info(ModelState* state);
