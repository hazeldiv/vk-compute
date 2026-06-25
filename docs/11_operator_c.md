# src/operator.c - Operator Registry and GPU Kernel Dispatch

=============================================================================
FILE: src/operator.c
PURPOSE: Registry of all GPU compute kernels. Compiles shaders, manages
         pipelines, and provides CPU-side wrappers that bind buffers and
         dispatch compute shaders.
=============================================================================

## Dependencies
- include/operator.h
- include/pipeline.h
- include/descriptor.h
- include/buffer.h
- include/dispatch.c (existing)

## Key Data Structures

-------------------------------------------------------------------------------
OpRegistry (defined in include/operator.h)
-------------------------------------------------------------------------------
  - device: Device*
  - pipelines[32]: One VkPipeline per operator type
  - pipeline_loaded[32]: bool flags
  - descriptor_layout: DescriptorLayout (shared layout for all ops)
  - shader_dir: char[256] - path to compiled .spv files

## Key Functions

-------------------------------------------------------------------------------
static Pipeline compile_shader(OpRegistry* reg, OpType type, const char* filename)
-------------------------------------------------------------------------------
  - Loads .spv from reg->shader_dir/filename
  - Creates VkShaderModule
  - Creates VkPipeline with the shader
  - Uses the shared descriptor layout
  - Returns Pipeline (wraps VkPipeline + VkPipelineLayout)
  - Returns empty Pipeline on failure

-------------------------------------------------------------------------------
static void build_shared_descriptor_layout(OpRegistry* reg)
-------------------------------------------------------------------------------
  - Creates a DescriptorLayout that supports ALL operators
  - Bindings:
      binding 0: qkv weights / general input  [STORAGE_BUFFER]
      binding 1: output buffer              [STORAGE_BUFFER]
      binding 2: attention cache / kvcache  [STORAGE_BUFFER]
      binding 3: scratch / temporary        [STORAGE_BUFFER]
      binding 4: auxiliary (norm weight, etc) [STORAGE_BUFFER]
      binding 5: auxiliary 2                 [STORAGE_BUFFER]
      binding 6: auxiliary 3                 [STORAGE_BUFFER]
      binding 7: auxiliary 4                 [STORAGE_BUFFER]
  - All bindings: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
  - All stages: VK_SHADER_STAGE_COMPUTE_BIT
  - This layout is shared across all shaders (each uses a subset)

-------------------------------------------------------------------------------
OpRegistry* op_registry_create(Device* device, const char* shader_dir)
-------------------------------------------------------------------------------
  PUBLIC API
  1. Allocate OpRegistry, set device and shader_dir
  2. Initialize all pipeline_loaded[] = false
  3. build_shared_descriptor_layout()
  4. Pre-compile the most common shaders eagerly:
      compile_shader(OP_GEMM, "gemm.spv")
      compile_shader(OP_RMSNORM, "rmsnorm.spv")
      compile_shader(OP_ATTENTION_FLASH, "attention_flash.spv")
      compile_shader(OP_ROPE, "rope.spv")
      compile_shader(OP_MLP, "mlp.spv")
      compile_shader(OP_SOFTMAX, "softmax.spv")
  5. Lazily compile others on first use (op_* wrapper calls compile if needed)
  6. Return OpRegistry*

-------------------------------------------------------------------------------
void op_registry_destroy(OpRegistry* registry)
-------------------------------------------------------------------------------
  1. Destroy each loaded pipeline
  2. Destroy descriptor layout
  3. Free(registry)

-------------------------------------------------------------------------------
static void ensure_pipeline_loaded(OpRegistry* reg, OpType type)
-------------------------------------------------------------------------------
  - If pipeline_loaded[type] is false:
      Call compile_shader() with the right filename
      Set pipeline_loaded[type] = true
  - This is called by all op_* wrappers for lazy compilation

-------------------------------------------------------------------------------
void op_dispatch(Pipeline* pipeline, CommandBuffer* cmd,
                 Buffer* bindings[], int num_bindings,
                 const void* push_const, int push_size,
                 int wg_x, int wg_y, int wg_z,
                 int gl_x, int gl_y, int gl_z)
-------------------------------------------------------------------------------
  PUBLIC (internal helper)
  1. vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->vk)
  2. For each binding i:
      vkCmdBindDescriptorSets(cmd, ..., 1, &desc_set, ...)
      (Note: descriptor sets are pre-allocated in InferenceContext)
  3. If push_const and push_size > 0:
      vkCmdPushConstants(cmd, pipeline_layout, ..., push_const, push_size)
  4. vkCmdDispatch(cmd,
        ceil(gl_x / wg_x), ceil(gl_y / wg_y), ceil(gl_z / wg_z))

-------------------------------------------------------------------------------
int op_rmsnorm(OpRegistry* registry, CommandBuffer* cmd, OpArgs_RMSNorm* args)
-------------------------------------------------------------------------------
  1. ensure_pipeline_loaded(registry, OP_RMSNORM)
  2. Get/allocate descriptor set
  3. Bind buffers:
      binding 0: args->input.buffer
      binding 1: args->output.buffer
      binding 4: args->weight.buffer (norm gamma)
  4. Push constants: {hidden_size, seq_len, eps}
  5. Dispatch: workgroup=128x1x1, global=seq_len*batch*hidden_size/128
  6. Return 0

-------------------------------------------------------------------------------
int op_rope(OpRegistry* registry, CommandBuffer* cmd, OpArgs_RoPE* args)
-------------------------------------------------------------------------------
  1. ensure_pipeline_loaded(registry, OP_ROPE)
  2. Bind buffers:
      binding 0: args->query.buffer
      binding 1: args->output_q.buffer
      binding 2: args->key.buffer
      binding 3: args->output_k.buffer
      binding 4: args->cos_table.buffer
      binding 5: args->sin_table.buffer
  3. Push constants: {seq_len, head_dim, num_heads, num_kv_heads, base}
  4. Dispatch: one thread per (batch, head, seq, dim_pair)
      workgroup = 128x1x1, global = batch * num_heads * seq_len * (head_dim/2)
  5. Return 0

-------------------------------------------------------------------------------
int op_attention_flash(OpRegistry* registry, CommandBuffer* cmd,
                       OpArgs_FlashAttention* args)
-------------------------------------------------------------------------------
  1. ensure_pipeline_loaded(registry, OP_ATTENTION_FLASH)
  2. Bind buffers:
      binding 0: args->query.buffer
      binding 1: args->key.buffer
      binding 2: args->value.buffer
      binding 3: args->output.buffer
      binding 4: args->key_cache.buffer (current layer)
      binding 5: args->value_cache.buffer (current layer)
      binding 6: args->attention_mask.buffer (or null)
  3. Push constants:
      {batch, num_heads, num_kv_heads, seq_len, head_dim,
       layer_idx, is_prefill, sliding_window}
  4. Dispatch Flash Attention kernel
      Workgroup: [tile_M, tile_N] e.g., 16x16
      Grid: [num_heads, num_blocks_M, num_blocks_K] for prefill
            or [num_heads, 1, 1] for single token decode
  5. Return 0

-------------------------------------------------------------------------------
int op_mlp(OpRegistry* registry, CommandBuffer* cmd, OpArgs_MLP* args)
-------------------------------------------------------------------------------
  1. ensure_pipeline_loaded(registry, OP_MLP)
  2. Bind buffers:
      binding 0: args->input.buffer
      binding 1: args->output.buffer (final result)
      binding 2: args->gate_weight.buffer
      binding 3: args->up_weight.buffer
      binding 4: args->down_weight.buffer
      binding 5: scratch.buffer (for intermediate gate*silu(up))
  3. Push constants:
      {batch, seq_len, hidden_size, intermediate_size}
  4. Dispatch MLP kernel
      Workgroup: 256x1x1
      Global: batch * seq_len * (intermediate_size / 256)
  5. Return 0

-------------------------------------------------------------------------------
int op_add_residual(OpRegistry* registry, CommandBuffer* cmd,
                    Tensor* input, Tensor* residual, Tensor* output)
-------------------------------------------------------------------------------
  1. ensure_pipeline_loaded(registry, OP_ADD)  (or reuse existing kernel)
  2. Bind 3 buffers to binding slots
  3. Push: {num_elements}
  4. Dispatch: 256 threads per element block
  5. Return 0

-------------------------------------------------------------------------------
int op_gemm(OpRegistry* registry, CommandBuffer* cmd,
            Buffer* a, Buffer* b, Buffer* c,
            int m, int n, int k,
            bool trans_a, bool trans_b)
-------------------------------------------------------------------------------
  - Delegates to existing compute.c GEMM implementation
  - Wraps the existing API for consistency with operator pattern
  - Could alternatively just call compute_gemm() directly in inference.c

## Shader Compilation Schedule

  EAGER (on op_registry_create):
    - OP_GEMM (gemm.spv) - needed immediately for embedding
    - OP_RMSNORM (rmsnorm.spv) - needed for every layer
    - OP_ATTENTION_FLASH (attention_flash.spv) - core operation
    - OP_ROPE (rope.spv) - needed for every layer
    - OP_MLP (mlp.spv) - needed for every layer

  LAZY (on first use):
    - OP_LAYERNORM (layernorm.spv)
    - OP_ATTENTION (attention.spv) - naive fallback
    - OP_SOFTMAX (softmax.spv) - rarely needed separately (fused)
    - OP_SILU (silu.spv) - fused into MLP
    - OP_QUANTIZE (quantize.spv)
    - OP_DEQUANTIZE (dequantize.spv)
