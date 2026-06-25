# include/tokenizer.h - Tokenization Interface

#pragma once
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
// Wraps a tokenizer implementation. Options:
//   1. HuggingFace tokenizer (requires linking HF libraries)
//   2. SentencePiece (C++ library, lighter weight)
//   3. Tiktoken (Python only - use FFI or subprocess)
//   4. Simple character-level tokenizer (demo only)
//
// For Phase 1-2, recommended: integrate SentencePiece or use
// Python subprocess + file-based IPC to avoid C complexity.
// The Tokenizer struct is an abstraction so the backend is swappable.
// ---------------------------------------------------------------------------

typedef struct Tokenizer Tokenizer;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * tokenizer_load()
 * Loads a tokenizer from a directory.
 * Looks for tokenizer.json, tokenizer_config.json, vocab files.
 *
 * @param tokenizer_dir  Path to tokenizer files
 * @return               Tokenizer* or NULL on failure
 */
Tokenizer* tokenizer_load(const char* tokenizer_dir);

/**
 * tokenizer_free()
 * Frees the tokenizer and all associated resources.
 */
void tokenizer_free(Tokenizer* tok);

/**
 * tokenizer_encode()
 * Converts text to token IDs.
 *
 * @param tok        Tokenizer
 * @param text       Input text (null-terminated)
 * @param out_tokens Output buffer (caller allocates)
 * @param max_tokens Size of output buffer
 * @return           Number of tokens written, or -1 on error
 *
 * Example:
 *   int32_t tokens[512];
 *   int n = tokenizer_encode(tok, "Hello, world!", tokens, 512);
 */
int tokenizer_encode(Tokenizer* tok, const char* text,
                     int32_t* out_tokens, int32_t max_tokens);

/**
 * tokenizer_decode()
 * Converts token IDs back to text.
 *
 * @param tok        Tokenizer
 * @param tokens     Input token IDs
 * @param num_tokens Number of tokens
 * @param out_text   Output buffer (caller allocates, must be large enough)
 * @param max_len    Size of output buffer
 * @return           Number of chars written, or -1 on error
 *
 * Example:
 *   char text[4096];
 *   tokenizer_decode(tok, tokens, n, text, sizeof(text));
 */
int tokenizer_decode(Tokenizer* tok, const int32_t* tokens, int32_t num_tokens,
                     char* out_text, int32_t max_len);

/**
 * tokenizer_vocab_size()
 * Returns the vocabulary size.
 */
int32_t tokenizer_vocab_size(Tokenizer* tok);

/**
 * tokenizer_eos_token()
 * Returns the EOS/end-of-sequence token ID.
 */
int32_t tokenizer_eos_token(Tokenizer* tok);

/**
 * tokenizer_bos_token()
 * Returns the BOS/beginning-of-sequence token ID.
 * Returns -1 if tokenizer has no BOS token.
 */
int32_t tokenizer_bos_token(Tokenizer* tok);

/**
 * tokenizer_pad_token()
 * Returns the PAD token ID.
 * Returns -1 if tokenizer has no PAD token.
 */
int32_t tokenizer_pad_token(Tokenizer* tok);

// ---------------------------------------------------------------------------
// Backend implementations (internal, switch by tokenizer_load flags)
// ---------------------------------------------------------------------------

// tokenizer_hf.c     - HuggingFace tokenizer backend
// tokenizer_sp.c    - SentencePiece backend
// tokenizer_tiktoken.c - Tiktoken (via Python subprocess)

// For now, implement tokenizer_simple.c as a placeholder:
//   - Byte-pair encoding with a small built-in vocabulary
//   - Useful for testing before integrating a real tokenizer
