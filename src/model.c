#include <stdio.h>
#include "model.h"

const int model_attn_layer[MODEL_LAYERS] = {
    0,1, 0,1, 0,1
};

const QuantType model_layer_q[MODEL_LAYERS] = {
    QUANT_FP16,QUANT_FP16,
    QUANT_INT8,QUANT_INT8,
    QUANT_INT4,QUANT_INT4
};

const QuantType model_ffn_q[MODEL_LAYERS] = {
    QUANT_FP16,QUANT_FP16,
    QUANT_INT8,QUANT_INT8,
    QUANT_INT4,QUANT_INT4
};

int model_is_attention(int layer) {
    return model_attn_layer[layer];
}

int model_attn_count(void) {
    int n = 0;
    for (int i = 0; i < MODEL_LAYERS; i++) n += model_attn_layer[i];
    return n;
}

const char* model_shader(const char* base, QuantType q) {
    static char buf[160];
    const char* suffix = (q == QUANT_FP16) ? "FP16" : (q == QUANT_INT8) ? "INT8" : "INT4";
    snprintf(buf, sizeof(buf), "%s-%s.spv", base, suffix);
    return buf;
}