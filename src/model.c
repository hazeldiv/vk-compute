#include <stdio.h>
#include "model.h"

const char* model_shader(const char* base, QuantType q) {
    static char buf[160];
    const char* suffix = (q == QUANT_FP16) ? "FP16" : (q == QUANT_INT8) ? "INT8" : "INT4";
    snprintf(buf, sizeof(buf), "%s-%s.spv", base, suffix);
    return buf;
}
