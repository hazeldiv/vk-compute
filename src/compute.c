#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "session.h"
#include "model.h"
#include "generate.h"

static model_config spec = {
    .name = "qwen3.5-9b",
    .layerCount = 32,
    .layers = {
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}}, // 1
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 2
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 3
        {.attn = {ATTENTION_FULL,  QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 4
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 5
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 6
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 7
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 8
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 9
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 10
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 11
        {.attn = {ATTENTION_FULL,  QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 12
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 13
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 14
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 15
        {.attn = {ATTENTION_FULL,  QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 16
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 17
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 18
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 19
        {.attn = {ATTENTION_FULL,  QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 20
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 21
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 22
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 23
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 24
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 25
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 26
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 27
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 28
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 29
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 30
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 31
        {.attn = {ATTENTION_FULL,  QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}}, // 32
    },
    .embedQ = QUANT_FP16,
    .lmHeadQ = QUANT_FP16,
};

void compute(int argc, char** argv) {
    int maxNewTokens = (argc > 2) ? atoi(argv[2]) : 128;

    FILE* pf = fopen("prompt.bin", "rb");
    if (!pf) {
        fprintf(stderr, "prompt.bin not found; run tools/tokenize.py first\n");
        return;
    }
    fseek(pf, 0, SEEK_END);
    long bytes = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    if (bytes <= 0 || (bytes % (long)sizeof(uint32_t)) != 0) {
        fprintf(stderr, "prompt.bin is empty or malformed\n");
        fclose(pf);
        return;
    }
    int nPrompt = (int)(bytes / (long)sizeof(uint32_t));
    uint32_t* prompt = (uint32_t*)malloc(bytes);
    if (fread(prompt, 1, bytes, pf) != (size_t)bytes) {
        fprintf(stderr, "failed to read prompt.bin\n");
        fclose(pf);
        free(prompt);
        return;
    }
    fclose(pf);
    printf("prompt: %d tokens, max new: %d\n", nPrompt, maxNewTokens);

    session s = createSession();
    generator* g = createGenerator(s, &spec, MODEL_PREFILL_CHUNK);

    runGenerate(g, prompt, nPrompt, maxNewTokens);

    destroyGenerator(g);
    destroySession(s);
    free(prompt);
}