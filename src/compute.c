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
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_FULL,  QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
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
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}},
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

static int argflag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

void memInfo(void) {
    session s = createSession();
    dumpMemoryInfo(s.dev.physicalDevice);
    destroySession(s);
}

static void emitToken(uint32_t token, void* ctx) {
    (void)ctx;
    fwrite(&token, sizeof(uint32_t), 1, stdout);
    fflush(stdout);
}

void serverMain(int argc, char** argv) {
    const char* weightDir = argval(argc, argv, "--weights", "../model/Qwen3.5-9B-weight");
    const char* vocabHead = argval(argc, argv, "--vocab-head", NULL);
    const char* vocabEmbed = argval(argc, argv, "--vocab-embed", NULL);
    int eos = atoi(argval(argc, argv, "--eos", "248044"));
    int maxCtx = atoi(argval(argc, argv, "--max-ctx", "32768"));
    int maxNew = atoi(argval(argc, argv, "--max-new", "128"));
    const char* dumpDir = argval(argc, argv, "--dump", NULL);
    int dumpLayers = atoi(argval(argc, argv, "--dump-layers", "0"));
    int verboseWeights = argflag(argc, argv, "--verbose-weights");
    int timing = argflag(argc, argv, "--timing");
    if (maxNew < 1) maxNew = 1;

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    if (timing) setTimingEnabled(1);
    session s = createSession();
    generator* g = createGenerator(s, &spec, MODEL_PREFILL_CHUNK, weightDir, vocabHead, vocabEmbed, eos, maxCtx, verboseWeights);
    if (dumpDir != NULL) {
        generatorSetDumpDir(g, dumpDir);
        generatorSetDumpLayers(g, dumpLayers);
    }

    uint32_t* prompt = NULL;
    int promptCap = 0;

    for (;;) {
        uint32_t n = 0;
        if (fread(&n, sizeof(uint32_t), 1, stdin) != 1) break;
        if (n == 0) break;

        sample_params sp = {0};
        uint32_t seed = 0;
        if (fread(&sp.temperature, sizeof(float), 1, stdin) != 1) break;
        if (fread(&sp.repPenalty, sizeof(float), 1, stdin) != 1) break;
        if (fread(&sp.penaltyLength, sizeof(uint32_t), 1, stdin) != 1) break;
        if (fread(&sp.topK, sizeof(uint32_t), 1, stdin) != 1) break;
        if (fread(&sp.topP, sizeof(float), 1, stdin) != 1) break;
        if (fread(&seed, sizeof(uint32_t), 1, stdin) != 1) break;
        if (sp.penaltyLength > MAX_PENALTY_LEN) sp.penaltyLength = MAX_PENALTY_LEN;
        if (seed == 0) seed = 1;

        if ((int)n > promptCap) {
            prompt = (uint32_t*)realloc(prompt, sizeof(uint32_t) * n);
            promptCap = (int)n;
        }
        if (fread(prompt, sizeof(uint32_t), n, stdin) != n) break;

        generatorSetSampling(g, &sp, seed);
        resetGenerator(g);
        generateTokens(g, prompt, (int)n, maxNew, emitToken, NULL);
        uint32_t sentinel = 0xFFFFFFFFu;
        fwrite(&sentinel, sizeof(uint32_t), 1, stdout);
        fflush(stdout);
    }

    free(prompt);
    destroyGenerator(g);
    destroySession(s);
    if (timing) closeTimingLog();
}