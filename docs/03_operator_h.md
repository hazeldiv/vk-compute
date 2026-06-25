# include/operator.h - GPU Kernel Registry and CPU-Side Wrappers

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "buffer.h"
#include "tensor.h"

// ---------------------------------------------------------------------------
// Operator types
// ---------------------------------------------------------------------------

typedef enum {
    OP_LAYERNORM,
    OP_RMSNORM,
    OP_ROPE,
    OP_ATTENTION,          // Naive O(n^2) attention
    OP_ATTENTION_FLASH,     // Flash Attention (memory-efficient)
    OP_MLP,
    OP_SILU,
    OP_SOFTMAX,
    OP_GEMM,               // Reuse existing
    OP_QUANTIZE,
    OP_DEQUANTIZE,
    OP_ADD,                // Element-wise addition (residual)
    OP_MUL,                // Element-wise multiplication
    OP_ROTATION,           // Complex-number rotation for RoPE
    OP_ROTARY_EMBED,       // Full rotary embedding (combine rope with q/k)
} OpType;

// ---------------------------------------------------------------------------
// Operator arguments - each operator has its own parameter struct
// ---------------------------------------------------------------------------

// --- LayerNorm arguments ---
typedef struct {
    Tensor input;          // [batch, seq, hidden_size]
    Tensor output;         // [batch, seq, hidden_size]
    Tensor weight;         // [hidden_size] Learnable gamma
    Tensor bias;            // [hidden_size] Learnable beta (optional, can be null)
    float eps;
} OpArgs_LayerNorm;

// --- RMSNorm arguments ---
typedef struct {
    Tensor input;          // [batch, seq, hidden_size]
    Tensor output;         // [batch, seq, hidden_size]
    Tensor weight;         // [hidden_size] Learnable gamma (no bias)
    float eps;
} OpArgs_RMSNorm;

// --- RoPE arguments ---
typedef struct {
    Tensor query;          // [batch, num_heads, seq, head_dim]
    Tensor key;            // [batch, num_kv_heads, seq, head_dim]
    Tensor output_q;      // [batch, num_heads, seq, head_dim]
    Tensor output_k;       // [batch, num_kv_heads, seq, head_dim]
    int32_t seq_len;
    int32_t head_dim;
    double base;           // RoPE theta parameter
    // Precomputed cos/sin tables passed as additional buffers
    Tensor cos_table;      // [seq_len, head_dim]
    Tensor sin_table;      // [seq_len, head_dim]
} OpArgs_RoPE;

// --- Attention arguments ---
typedef struct {
    // Q, K, V as separate tensors
    Tensor query;          // [batch, num_heads, seq, head_dim]
    Tensor key;            // [batch, num_kv_heads, seq, head_dim]
    Tensor value;          // [batch, num_kv_heads, seq, head_dim]
    Tensor output;         // [batch, num_heads, seq, head_dim]

    // KV Cache (for autoregressive decoding)
    Tensor key_cache;      // [num_layers][batch, num_kv_heads, max_seq, head_dim]
    Tensor value_cache;    // [num_layers][batch, num_kv_heads, max_seq, head_dim]
    int32_t layer_idx;     // Which layer this attention is for

    // Optional: attention mask (for prefill)
    Tensor attention_mask; // [batch, 1, seq, seq] or null

    // Optional: sliding window
    int32_t sliding_window; // -1 if no sliding window

    bool is_prefill;       // true = processing full sequence, false = single token
} OpArgs_Attention;

// --- Flash Attention arguments ---
// Same as OpArgs_Attention but dispatching to the Flash Attention shader
// which computes attention in tiles to avoid O(n^2) memory.
typedef OpArgs_Attention OpArgs_FlashAttention;

// --- MLP arguments ---
typedef struct {
    Tensor input;          // [batch, seq, hidden_size]
    Tensor output;         // [batch, seq, hidden_size]
    Tensor gate_weight;    // [intermediate_size, hidden_size]
    Tensor up_weight;      // [intermediate_size, hidden_size]
    Tensor down_weight;    // [hidden_size, intermediate_size]
    // Activation is typically fused: gate * SiLU(up)
    // Output = down(gate * SiLU(up))
} OpArgs_MLP;

// --- Softmax arguments ---
typedef struct {
    Tensor input;          // [batch, num_heads, seq, seq] attention scores
    Tensor output;         // [batch, num_heads, seq, seq]
    Tensor attn_mask;      // Optional mask, or null
    int32_t axis;          // Which axis to softmax over (typically -1 = last dim)
} OpArgs_Softmax;

// --- Quantize/DeQuantize arguments ---
typedef struct {
    Tensor input;          // FP16/FP32 input
    Tensor output;         // INT8/INT4 quantized output
    Tensor scale;          // Quantization scale tensor
    // For INT4: packs 2 values per byte
} OpArgs_Quantize;

// ---------------------------------------------------------------------------
// OpRegistry - central registry of all compute pipelines
// ---------------------------------------------------------------------------
// The registry holds one VkPipeline per operator type.
// Pipelines are built once at model load time and reused forever.
// ---------------------------------------------------------------------------

typedef struct {
    // Device this registry is for
    Device* device;

    // One pipeline per operator type
    // Keyed by OpType enum value
    Pipeline pipelines[32];
    bool pipeline_loaded[32];

    // Descriptor set layout (shared by all operators that take buffers)
    // Each operator uses a subset of bindings.
    DescriptorLayout descriptor_layout;

    // Shader directories
    char shader_dir[256];
} OpRegistry;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * op_registry_create()
 * Creates the operator registry and compiles all shader pipelines.
 * Called once during engine initialization.
 *
 * @param device       Vulkan device
 * @param shader_dir   Path to compiled .spv shader files
 * @return             OpRegistry* or NULL on failure
 */
OpRegistry* op_registry_create(Device* device, const char* shader_dir);

/**
 * op_registry_destroy()
 * Destroys all pipelines and frees the registry.
 */
void op_registry_destroy(OpRegistry* registry);

/**
 * op_layernorm()
 * Executes LayerNorm on the GPU.
 *
 * @param registry     Operator registry
 * @param cmd          Command buffer (must be in recording state)
 * @param args         LayerNorm arguments
 * @return             0 on success
 */
int op_layernorm(OpRegistry* registry, CommandBuffer* cmd, OpArgs_LayerNorm* args);

/**
 * op_rmsnorm()
 * Executes RMSNorm on the GPU.
 * RMSNorm = input * (rms + eps)^(-1/2) where rms = sqrt(mean(x^2))
 * No bias term. Used by Qwen, Llama.
 */
int op_rmsnorm(OpRegistry* registry, CommandBuffer* cmd, OpArgs_RMSNorm* args);

/**
 * op_rope()
 * Executes Rotary Position Embedding on the GPU.
 * Applies rotation to query and key in frequency domain.
 * Handles half-precision (FP16) rotation efficiently.
 */
int op_rope(OpRegistry* registry, CommandBuffer* cmd, OpArgs_RoPE* args);

/**
 * op_attention()
 * Executes standard scaled dot-product attention.
 * WARNING: O(seq_len^2) memory. Use for short sequences or testing only.
 */
int op_attention(OpRegistry* registry, CommandBuffer* cmd, OpArgs_Attention* args);

/**
 * op_attention_flash()
 * Executes Flash Attention algorithm.
 * O(seq_len) memory instead of O(seq_len^2).
 * RECOMMENDED for production.
 */
int op_attention_flash(OpRegistry* registry, CommandBuffer* cmd,
                       OpArgs_FlashAttention* args);

/**
 * op_mlp()
 * Executes the feed-forward network:
 *   gate_proj(x) -> gate
 *   up_proj(x)  -> up
 *   act = silu(gate) * up
 *   down_proj(act) -> output
 *
 * The gate * SiLU(up) multiplication is fused in the shader.
 */
int op_mlp(OpRegistry* registry, CommandBuffer* cmd, OpArgs_MLP* args);

/**
 * op_softmax()
 * Executes softmax over the specified axis.
 * Typically softmax(axis=-1) on attention scores.
 */
int op_softmax(OpRegistry* registry, CommandBuffer* cmd, OpArgs_Softmax* args);

/**
 * op_add_residual()
 * Performs: output = input + residual
 * Fused element-wise addition for residual connections.
 * Fused add avoids an extra VRAM round-trip.
 */
int op_add_residual(OpRegistry* registry, CommandBuffer* cmd,
                    Tensor* input, Tensor* residual, Tensor* output);

/**
 * op_dequantize()
 * Converts INT8/INT4 weights back to FP16 for GEMM compute.
 */
int op_dequantize(OpRegistry* registry, CommandBuffer* cmd,
                  OpArgs_Quantize* args);

/**
 * op_quantize()
 * Converts FP16 activations to INT8 for quantized GEMM.
 */
int op_quantize(OpRegistry* registry, CommandBuffer* cmd,
                OpArgs_Quantize* args);

// ---------------------------------------------------------------------------
// Internal helper (used by operators to dispatch)
// ---------------------------------------------------------------------------

/**
 * op_dispatch()
 * Internal: binds descriptors and dispatches a compute shader.
 * Used by all operators internally.
 *
 * @param pipeline     Which pipeline to dispatch
 * @param cmd           Command buffer
 * @param bindings      Array of buffer pointers
 * @param num_bindings  How many buffers
 * @param push_const    Push constant data (up to 128 bytes)
 * @param push_size     Size of push constant data
 * @param workgroup_x   X dimension
 * @param workgroup_y   Y dimension
 * @param workgroup_z   Z dimension
 * @param global_x      Total X workitems
 * @param global_y      Total Y workitems
 * @param global_z      Total Z workitems
 */
void op_dispatch(Pipeline* pipeline, CommandBuffer* cmd,
                 Buffer* bindings[], int num_bindings,
                 const void* push_const, int push_size,
                 int workgroup_x, int workgroup_y, int workgroup_z,
                 int global_x, int global_y, int global_z);
