# src/engine.c - Top-Level Facade Implementation

=============================================================================
FILE: src/engine.c
PURPOSE: The public entry point. Owns everything. Simple API hiding all
         complexity. This is what main.c includes.
=============================================================================

## Dependencies
- include/engine.h
- include/device.h
- include/container.h
- include/inference.h
- include/tokenizer.h
- include/command.h

## Key Functions

-------------------------------------------------------------------------------
LLMEngine* llm_engine_create(EngineConfig* config)
-------------------------------------------------------------------------------
  PUBLIC API - the ONE function to rule them all
  1. Allocate LLMEngine struct and zero-initialize
  2. Copy config: engine->config = *config

  3. Initialize Vulkan device
     if (config->device_name != NULL):
         device_init_named(&engine->device, config->device_name)
     else:
         device_init_first(&engine->device)  // Pick first available

  4. Create command pool
     command_pool_create(&engine->cmd_pool, &engine->device)

  5. Load tokenizer
     if (config->tokenizer_path != NULL):
         engine->tokenizer = tokenizer_load(config->tokenizer_path)
     else:
         engine->tokenizer = tokenizer_load(config->model_path)
     If tokenizer fails, print warning but continue (can still run on token IDs)

  6. Create inference context
     engine->inference = inference_context_create(
         config->model_path,
         &engine->device,
         &engine->cmd_pool,
         config->max_seq_len > 0 ? config->max_seq_len : 2048
     )

  7. If config->verbose:
     llm_engine_print_info(engine)
     llm_engine_get_stats() -> print VRAM usage

  8. Return engine*

  Error handling: if any step fails, clean up what was created and return NULL

-------------------------------------------------------------------------------
void llm_engine_destroy(LLMEngine* engine)
-------------------------------------------------------------------------------
  1. if (engine->inference) inference_context_destroy(engine->inference)
  2. if (engine->tokenizer) tokenizer_free(engine->tokenizer)
  3. command_pool_destroy(&engine->cmd_pool)
  4. device_destroy(&engine->device)
  5. free(engine)

-------------------------------------------------------------------------------
int llm_engine_generate(LLMEngine* engine,
                        const char* prompt,
                        char* output_text,
                        int32_t max_len,
                        float temperature,
                        int32_t max_tokens)
-------------------------------------------------------------------------------
  PUBLIC API - simple blocking generation

  1. Encode prompt -> token_ids
     n = tokenizer_encode(engine->tokenizer, prompt, tokens, MAX_TOKENS)
     If tokenizer failed, return -1

  2. Set up sampling params
     SamplingParams params = {
         .type = temperature > 0.0f ? 1 : 0,  // 0=greedy, 1=temp
         .temperature = temperature,
         .top_k = engine->config.default_top_k,
         .top_p = engine->config.default_top_p,
         .eos_token_id = tokenizer_eos_token(engine->tokenizer),
         .max_new_tokens = max_tokens,
         .min_new_tokens = 1,
     }

  3. Reset KV cache
     inference_reset_kv_cache(engine->inference)

  4. Run generation
     GenerationResult result = {0}
     inference_generate(engine->inference, tokens, n, &params, &result)

  5. Decode output tokens
     tokenizer_decode(engine->tokenizer, result.token_ids, result.num_tokens,
                      output_text, max_len)

  6. Free result buffers
     free(result.token_ids)

  7. Return number of chars written

-------------------------------------------------------------------------------
int llm_engine_generate_streaming(LLMEngine* engine,
                                   const char* prompt,
                                   TokenCallback callback,
                                   void* user_data,
                                   float temperature,
                                   int32_t max_tokens)
-------------------------------------------------------------------------------
  PUBLIC API - streaming generation with callback per token

  1. Encode prompt
  2. Prefill phase (encode all prompt tokens once):
     inference_forward(ctx, tokens, n, NULL)
     // KV cache now populated

  3. Decode loop:
     while (generated < max_tokens):
         // Get logits for current position
         inference_forward(ctx, &last_token, 1, logits)
         last_token = sample_token(logits, ...)
         if (last_token == EOS) break

         // Call callback with token
         char token_str[32];
         tokenizer_decode_one(engine->tokenizer, last_token, token_str, 32);
         if (!callback(last_token, token_str, user_data))
             break;  // User requested stop

         generated++

  4. Return 0

-------------------------------------------------------------------------------
int llm_engine_chat(LLMEngine* engine,
                    const char* system_prompt,
                    const char* user_message,
                    char* output,
                    int32_t max_len,
                    float temperature,
                    int32_t max_tokens)
-------------------------------------------------------------------------------
  PUBLIC API - convenient chat interface

  1. Build prompt in ChatML format:
     <|im_start|>system
     {system_prompt}<|im_end|>
     <|im_start|>user
     {user_message}<|im_end|>
     <|im_start|>assistant

  2. Encode the full prompt
  3. Generate (with EOS = <|im_end|>)
  4. Extract assistant response from output (between <|im_start|>assistant
     and <|im_end|>)

  Returns number of chars in the assistant's response only.

-------------------------------------------------------------------------------
void llm_engine_get_stats(LLMEngine* engine, EngineStats* out_stats)
-------------------------------------------------------------------------------
  - Query Vulkan for VRAM usage (vkGetDeviceMemoryCommitment or tracking)
  - Compute model stats from ModelState
  - Fill out_stats struct

-------------------------------------------------------------------------------
void llm_engine_reset(LLMEngine* engine)
-------------------------------------------------------------------------------
  - inference_reset_kv_cache(engine->inference)

-------------------------------------------------------------------------------
void llm_engine_print_info(LLMEngine* engine)
-------------------------------------------------------------------------------
  - Print model name, architecture summary
  - Print layer count, head count, hidden size
  - Print parameter count
  - Print dtype
  - Print VRAM usage

## Example main.c (what it looks like to use the engine)

    #include "engine.h"

    int main() {
        EngineConfig config = {
            .model_path = "model/Qwen3.5-9B",
            .max_seq_len = 4096,
            .default_temperature = 0.7f,
            .default_max_tokens = 512,
            .verbose = true,
        };

        LLMEngine* engine = llm_engine_create(&config);
        if (!engine) {
            printf("Failed to create engine\n");
            return 1;
        }

        char output[8192];
        llm_engine_generate(engine,
            "Explain how a transformer architecture works:",
            output, sizeof(output),
            0.7f, 512);

        printf("%s\n", output);

        llm_engine_destroy(engine);
        return 0;
    }
