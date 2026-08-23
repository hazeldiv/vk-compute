#include <stdint.h>
#include "session.h"
#include "model.h"
#include "generate.h"

static model_config spec = {
    .name = "qwen3.5-9b",
    .layerCount = 32,
    .layers = {
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}}, // 1
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}}, // 2
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 3
        {.attn = {ATTENTION_FULL,  QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 4
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 5
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 6
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 7
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
        {.attn = {ATTENTION_FULL,  QUANT_INT4}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 24
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT4}}, // 25
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 26
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 27
        {.attn = {ATTENTION_FULL,  QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 28
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 29
        {.attn = {ATTENTION_DELTA, QUANT_INT8}, .ffn = {FFN_SWIGLU, QUANT_INT8}}, // 30
        {.attn = {ATTENTION_DELTA, QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}}, // 31
        {.attn = {ATTENTION_FULL,  QUANT_FP16}, .ffn = {FFN_SWIGLU, QUANT_FP16}}, // 32
    },
    .embedQ = QUANT_FP16,
    .lmHeadQ = QUANT_FP16,
};

void compute() {
    session s = createSession();
    generator g = createGenerator(s, &spec, MODEL_MAX_GEMM);

    uint32_t prompt[MODEL_MAX_GEMM];
    for (int i = 0; i < MODEL_MAX_GEMM; i++) {
        prompt[i] = (uint32_t)((i * 1237 + 555) % MODEL_VOCAB);
    }

    runGenerate(&g, prompt, MODEL_MAX_GEMM, 1024);

    destroyGenerator(&g);
    destroySession(s);
}