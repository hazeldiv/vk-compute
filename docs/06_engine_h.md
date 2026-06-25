# include/engine.h - Top-Level LLMEngine Facade

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "device.h"
#include "container.h"
#include "inference.h"
#include "tokenizer.h"

// ---------------------------------------------------------------------------
// LLMEngine - the main entry point
// ---------------------------------------------------------------------------
// A single struct that owns everything: device, model, tokenizer, inference.
// This is the only header the end user (main.c) needs to include.
//
// Design: simple interface, all complexity is hidden inside.
// ---------------------------------------------------------------------------

typedef struct LLMEngine LLMEngine;

typedef struct {
    // Required
    const char* model_path;     // Path to model directory

    // Vulkan
    const char* device_name;    // Device name substring to match, or NULL for first

    // Memory
    int32_t max_seq_len;        // KV cache max sequence length
    int32_t max_batch_size;     // Batch size (default: 1)

    // Tokenizer (optional, auto-detected if NULL)
    const char* tokenizer_path; // Path to tokenizer dir, or NULL to use model_path

    // Quantization
    bool quantize_weights;       // Quantize weights to INT8 on load
    bool quantize_activations;   // Quantize activations during compute

    // Generation defaults
    float default_temperature;
    int32_t default_max_tokens;
    int32_t default_top_k;
    float default_top_p;

    // Logging
    bool verbose;
    void (*log_callback)(const char* msg);  // Optional logging function
} EngineConfig;

// ---------------------------------------------------------------------------
// Extended statistics
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t vram_used;
    uint64_t vram_total;
    uint64_t weights_size_bytes;
    uint64_t kv_cache_size_bytes;
    double   throughput_tokens_per_sec;
    double   avg_time_per_token_ms;
    int32_t  num_layers;
    int32_t  num_params;      // Approximate parameter count in billions
} EngineStats;

// ---------------------------------------------------------------------------
// Generation output callback
// ---------------------------------------------------------------------------
// Called for each generated token (streamed output).
// Return false to stop generation early.
// ---------------------------------------------------------------------------

typedef bool (*TokenCallback)(int32_t token_id, const char* token_str, void* user_data);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * llm_engine_create()
 * Creates and initializes the LLM engine.
 * This is the ONLY function most users need to call.
 *
 * @param config  Engine configuration
 * @return        LLMEngine* or NULL on failure
 */
LLMEngine* llm_engine_create(EngineConfig* config);

/**
 * llm_engine_destroy()
 * Destroys the engine and frees all resources.
 */
void llm_engine_destroy(LLMEngine* engine);

/**
 * llm_engine_generate()
 * Generates text from a prompt.
 * Blocks until generation is complete.
 *
 * @param engine       LLM engine
 * @param prompt       Input prompt text
 * @param output_text  Output buffer (caller allocates)
 * @param max_len      Size of output buffer in bytes
 * @param temperature  Sampling temperature (0.0 = greedy)
 * @param max_tokens   Maximum new tokens to generate
 * @return             Number of characters written, or -1 on error
 */
int llm_engine_generate(LLMEngine* engine,
                        const char* prompt,
                        char* output_text,
                        int32_t max_len,
                        float temperature,
                        int32_t max_tokens);

/**
 * llm_engine_generate_streaming()
 * Like llm_engine_generate but calls callback for each token.
 *
 * @param engine       LLM engine
 * @param prompt       Input prompt
 * @param callback     Called with each token
 * @param user_data    Passed to callback
 * @param temperature  Sampling temperature
 * @param max_tokens   Maximum tokens to generate
 * @return             0 on success, -1 on error
 */
int llm_engine_generate_streaming(LLMEngine* engine,
                                   const char* prompt,
                                   TokenCallback callback,
                                   void* user_data,
                                   float temperature,
                                   int32_t max_tokens);

/**
 * llm_engine_chat()
 * Convenience: chat completion with a system prompt and message history.
 * Supports multi-turn conversations.
 *
 * Message format: array of {"role": "user"|"assistant"|"system", "content": "..."}
 * This is a simplified version. For production, implement the ChatML format.
 *
 * @param messages     Array of message structs
 * @param num_messages Number of messages
 * @param output       Output buffer
 * @param max_len      Buffer size
 * @param temperature  Temperature
 * @param max_tokens   Max new tokens
 */
int llm_engine_chat(LLMEngine* engine,
                    const char* system_prompt,
                    const char* user_message,
                    char* output,
                    int32_t max_len,
                    float temperature,
                    int32_t max_tokens);

/**
 * llm_engine_get_stats()
 * Returns engine statistics (memory usage, performance).
 */
void llm_engine_get_stats(LLMEngine* engine, EngineStats* out_stats);

/**
 * llm_engine_reset()
 * Resets the KV cache and internal state.
 * Call between separate generation requests.
 */
void llm_engine_reset(LLMEngine* engine);

/**
 * llm_engine_get_model_info()
 * Prints model architecture info to stdout.
 */
void llm_engine_print_info(LLMEngine* engine);

// ---------------------------------------------------------------------------
// Internal structure (not exposed to public)
// Defined in engine.c
// ---------------------------------------------------------------------------

struct LLMEngine {
    EngineConfig    config;
    Device          device;
    CommandPool     cmd_pool;
    InferenceContext* inference;
    Tokenizer*      tokenizer;
    // ... more internal state
};
