# src/model.c - Model Loading and Weight Management

=============================================================================
FILE: src/model.c
PURPOSE: Loads model weights from disk (safetensors) into VRAM buffers.
         Manages the ModelState lifecycle.
=============================================================================

## Dependencies
- include/model.h
- include/buffer.h
- include/tensor.h
- External: safetensors parsing (use a single-header library like stb style)

## Key Functions

-------------------------------------------------------------------------------
static int parse_config_json(const char* dir_path, ModelConfig* out_config)
-------------------------------------------------------------------------------
  - Reads model_dir/config.json
  - Parses JSON fields into ModelConfig struct
  - Maps HuggingFace-style config names to internal structs
  - Handles Qwen-specific config fields
  - Returns 0 on success

-------------------------------------------------------------------------------
static int load_embeddings(const char* dir_path, Device* device, Embeddings* out)
-------------------------------------------------------------------------------
  - Loads embed_tokens from model.safetensors (key: "model.embed_tokens.weight")
  - Loads lm_head from model.safetensors (key: "lm_head.weight")
    or shares embed_tokens if no separate head
  - Creates VRAM buffers for each
  - Uploads weight data via staging buffers
  - out->embed_tokens: Buffer [vocab_size, hidden_size], MEMORY_VRAM
  - out->lm_head: Buffer [vocab_size, hidden_size], MEMORY_VRAM
  - Returns 0 on success

-------------------------------------------------------------------------------
static int load_layer_weights(const char* dir_path, Device* device,
                              int32_t layer_idx, LayerWeights* out)
-------------------------------------------------------------------------------
  - Opens model.safetensors and reads only the tensors for this layer
  - Expected keys (Qwen format):
      model.layers.{layer}.self_attn.q_proj.weight
      model.layers.{layer}.self_attn.k_proj.weight
      model.layers.{layer}.self_attn.v_proj.weight
      model.layers.{layer}.self_attn.o_proj.weight
      model.layers.{layer}.mlp.gate_proj.weight
      model.layers.{layer}.mlp.up_proj.weight
      model.layers.{layer}.mlp.down_proj.weight
      model.layers.{layer}.input_layernorm.weight
      model.layers.{layer}.post_attention_layernorm.weight
  - Creates VRAM buffers for each
  - Converts dtype if needed (safetensors may be FP16, BF16, INT8, etc.)
  - Returns 0 on success

-------------------------------------------------------------------------------
static int allocate_kv_cache(Device* device, ModelConfig* config,
                              int32_t max_seq_len, KVCache* out)
-------------------------------------------------------------------------------
  - Pre-allocates key_cache[layer]: Buffer [num_kv_heads, max_seq_len, head_dim]
  - Pre-allocates value_cache[layer]: same shape
  - out->max_seq_len = max_seq_len
  - out->current_seq_len = 0
  - Returns 0 on success

-------------------------------------------------------------------------------
static int precompute_rope_tables(ModelConfig* config, Device* device,
                                   int32_t max_seq_len, Buffer* cos_out,
                                   Buffer* sin_out)
-------------------------------------------------------------------------------
  - Computes RoPE cos/sin tables up to max_seq_len
  - Formula: theta_i = base * (base_ratio ^ (i/head_dim))
  - For each position p and dimension d:
      cos_table[p,d] = cos(p * theta_d)
      sin_table[p,d] = sin(p * theta_d)
  - Uploads to VRAM
  - Returns 0 on success

-------------------------------------------------------------------------------
static uint64_t estimate_vram_usage(ModelConfig* config, int32_t max_seq_len)
-------------------------------------------------------------------------------
  - Estimates total VRAM needed:
      weights = num_layers * (qkv + o + gate + up + down + 2*norm)
             + embeddings + lm_head
      kv_cache = num_layers * num_kv_heads * max_seq_len * head_dim * 2
      activations = scratch space estimate
  - Returns bytes needed

-------------------------------------------------------------------------------
int model_load_from_directory(const char* dir_path, Device* device,
                               ModelState** out_state)
-------------------------------------------------------------------------------
  PUBLIC API
  1. Allocate ModelState*
  2. Parse config.json into state->config
  3. Validate config (check required fields)
  4. Estimate VRAM usage, check available memory
  5. Allocate KV cache buffers
  6. Precompute RoPE tables
  7. Allocate scratch buffers
  8. Load embeddings into VRAM
  9. For each layer:
        load_layer_weights() for this layer
  10. Build descriptor layout for operators
  11. Return populated ModelState*

-------------------------------------------------------------------------------
void model_unload(ModelState* state)
-------------------------------------------------------------------------------
  PUBLIC API
  1. Destroy KV cache buffers (free(state->kv_cache.key_cache), etc.)
  2. Destroy scratch buffers
  3. For each layer: destroy all LayerWeights buffers
  4. Destroy embedding buffers
  5. Free RoPE table buffers
  6. Free(state)

-------------------------------------------------------------------------------
void model_print_info(ModelState* state)
-------------------------------------------------------------------------------
  PUBLIC API
  - Prints: model name, num_layers, hidden_size, num_heads, vocab_size
  - Prints: dtype, VRAM usage breakdown, KV cache size
  - Prints: parameter count estimate

## Data Flow

    disk (safetensors)                    VRAM (GPU)
    ================                     ==============
    config.json  ──parse──> ModelConfig
    model.safetensors
      ├── embed_tokens ──load──> state->embeddings.embed_tokens [VRAM]
      ├── lm_head      ──load──> state->embeddings.lm_head [VRAM]
      └── layers[i]
           ├── q_proj ──load──> state->layers[i].q_proj [VRAM]
           ├── k_proj ──load──> state->layers[i].k_proj [VRAM]
           ├── v_proj ──load──> state->layers[i].v_proj [VRAM]
           ├── o_proj ──load──> state->layers[i].o_proj [VRAM]
           ├── gate    ──load──> state->layers[i].gate_proj [VRAM]
           ├── up      ──load──> state->layers[i].up_proj [VRAM]
           ├── down    ──load──> state->layers[i].down_proj [VRAM]
           ├── ln1     ──load──> state->layers[i].input_layernorm [VRAM]
           └── ln2     ──load──> state->layers[i].post_attention_layernorm [VRAM]

    KVCache pre-allocated ───────────────> state->kv_cache.key_cache[i] [VRAM]
                                          state->kv_cache.value_cache[i] [VRAM]

    RoPE tables computed ─────────────────> cos_table [VRAM], sin_table [VRAM]
