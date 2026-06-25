# src/tokenizer.c - Tokenization Implementation

=============================================================================
FILE: src/tokenizer.c
PURPOSE: Tokenizer implementations. This file has multiple backends.
         For Phase 1, use a simple subprocess-based approach to avoid
         C complexity. For Phase 2, integrate SentencePiece.
=============================================================================

## Implementation Approaches (in order of complexity)

### APPROACH A: Python Subprocess (RECOMMENDED FOR PHASE 1)
  - Keep your tokenizer logic in Python
  - C code calls `python tokenizer.py encode "text"` via pipe
  - Pros: Zero C complexity, full tokenizer support immediately
  - Cons: Overhead per encode/decode call (~5-20ms)
  - Good enough for: Testing, prototyping, batch inference
  - Example:

    // tokenizer_subprocess.c
    static int encode_subprocess(Tokenizer* tok, const char* text,
                                  int32_t* tokens, int32_t max_tokens) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "python tokenizer.py encode \"%s\"", text);

        FILE* fp = popen(cmd, "r");
        if (!fp) return -1;

        // Read token IDs from stdout, one per line
        int count = 0;
        char line[32];
        while (fgets(line, sizeof(line), fp) && count < max_tokens) {
            tokens[count++] = atoi(line);
        }
        pclose(fp);
        return count;
    }

### APPROACH B: SentencePiece (C++ library, Phase 2)
  - Link against sentencepiece library
  - Load .model file
  - Much faster, no subprocess overhead
  - Requires: libsentencepiece-dev or bundled source

### APPROACH C: Bundled minimal tokenizer (Phase 3)
  - Implement a minimal BPE tokenizer in pure C
  - Only supports the specific model's vocabulary
  - Eliminates all external dependencies

## Tokenizer Struct (opaque, backend-specific)

struct Tokenizer {
    char type[16];  // "subprocess", "sentencepiece", "simple"

    // Backend-specific data
    union {
        struct { char script_path[256]; } subprocess;
        struct { void* sp_model; } sentencepiece;
        struct { void* model_data; } simple;
    } backend;

    int32_t vocab_size;
    int32_t eos_token_id;
    int32_t bos_token_id;
    int32_t pad_token_id;
    char eos_str[16];
    char bos_str[16];
};

## Key Functions

Tokenizer* tokenizer_load(const char* tokenizer_dir)
  1. Look for tokenizer files in directory
  2. Try to detect type:
       if tokenizer.py exists: use APPROACH A (subprocess)
       if tokenizer.model exists: use APPROACH B (sentencepiece)
       else: return NULL (no tokenizer found)
  3. Initialize backend
  4. Read vocab size, EOS/BOS/PAD token IDs from config
  5. Return Tokenizer*

int tokenizer_encode(Tokenizer* tok, const char* text,
                     int32_t* out_tokens, int32_t max_tokens)
  1. Dispatch to backend:
       tok->type == "subprocess" -> encode_subprocess()
       tok->type == "sentencepiece" -> encode_sentencepiece()
       tok->type == "simple" -> encode_simple()
  2. Return number of tokens, or -1 on error

int tokenizer_decode(Tokenizer* tok, const int32_t* tokens, int32_t num_tokens,
                     char* out_text, int32_t max_len)
  1. Dispatch to backend
  2. Convert token IDs to text string
  3. Return number of characters, or -1 on error

void tokenizer_free(Tokenizer* tok)
  - Free backend-specific data
  - free(tok)

## Subprocess Script (tokenizer.py)

  #!/usr/bin/env python3
  # tokenizer.py - Simple tokenizer wrapper
  # Can use HuggingFace transformers or tiktoken

  import sys
  import json

  def load_tokenizer():
      from transformers import AutoTokenizer
      # Auto-detect from parent dir
      return AutoTokenizer.from_pretrained(".", trust_remote_code=True)

  tok = load_tokenizer()

  cmd = sys.argv[1]  # "encode" or "decode"
  if cmd == "encode":
      text = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip()
      tokens = tok.encode(text, return_tensors="pt")[0].tolist()
      for t in tokens:
          print(t)
  elif cmd == "decode":
      tokens = [int(x) for x in sys.stdin.read().split()]
      text = tok.decode(tokens)
      print(text, end="")

  # Note: for faster startup, keep tokenizer in memory between calls
  # by using a persistent server (see tokenizer_server.py)

## Server-based Subprocess (faster for many calls)

  For interactive use, the subprocess-per-call overhead is too high.
  Use a persistent server instead:

  1. tokenizer_server.py starts a long-running Python process
  2. Listens on stdin/stdout with a simple protocol:
       REQUEST: {"id": 1, "method": "encode", "text": "hello"}
       RESPONSE: {"id": 1, "result": [123, 456, 789]}

  3. C code opens pipe once, sends requests, reads responses
  4. Eliminates Python startup overhead per call (~100ms -> ~0.1ms)
