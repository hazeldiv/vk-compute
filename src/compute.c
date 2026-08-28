#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>
#include "session.h"
#include "model.h"
#include "generate.h"

static model_config spec = {
    .name = "qwen3.5-9b",
    .layerCount = 32,
    .layers = {
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}},
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_FULL,  QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_FULL,  QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_FULL,  QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_FULL,  QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_FULL,  QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}},
    },
    .embedQ = QUANT_FP16,
    .lmHeadQ = QUANT_FP16,
};

static const char* argval(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

void memInfo(void) {
    session s = createSession();
    dumpMemoryInfo(s.dev.physicalDevice);
    destroySession(s);
}

void serverMain(int argc, char** argv) {
    const char* weightDir = argval(argc, argv, "--weights", "../model/Qwen3.5-9B-weight");
    const char* vocabHead = argval(argc, argv, "--vocab-head", NULL);
    const char* vocabEmbed = argval(argc, argv, "--vocab-embed", NULL);
    int eos = atoi(argval(argc, argv, "--eos", "248044"));
    int maxCtx = atoi(argval(argc, argv, "--max-ctx", "32768"));
    int maxNew = atoi(argval(argc, argv, "--max-new", "128"));
    if (maxNew < 1) maxNew = 1;

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    session s = createSession();
    generator* g = createGenerator(s, &spec, MODEL_PREFILL_CHUNK, weightDir, vocabHead, vocabEmbed, eos, maxCtx);

    uint32_t* prompt = NULL;
    int promptCap = 0;
    uint32_t* out = (uint32_t*)malloc(sizeof(uint32_t) * maxNew);

    for (;;) {
        uint32_t n = 0;
        if (fread(&n, sizeof(uint32_t), 1, stdin) != 1) break;
        if (n == 0) break;
        if ((int)n > promptCap) {
            prompt = (uint32_t*)realloc(prompt, sizeof(uint32_t) * n);
            promptCap = (int)n;
        }
        if (fread(prompt, sizeof(uint32_t), n, stdin) != n) break;

        resetGenerator(g);
        int count = 0;
        generateTokens(g, prompt, (int)n, maxNew, out, &count);

        uint32_t m = (uint32_t)count;
        fwrite(&m, sizeof(uint32_t), 1, stdout);
        fwrite(out, sizeof(uint32_t), count, stdout);
        fflush(stdout);
    }

    free(prompt);
    free(out);
    destroyGenerator(g);
    destroySession(s);
}