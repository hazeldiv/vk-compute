#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "weights.h"

static tensor createTensor(session s, int seed, int rows, int cols, QuantType q, float wscale) {
    tensor t = {0};
    t.q = q;
    t.rows = rows;
    t.cols = cols;
    int blocks = (cols + 255) / 256;

    if (q == QUANT_FP16) {
        uint16_t* w = getDataFP16(seed, rows, cols);
        if (wscale != 1.0f) {
            for (int i = 0; i < rows * cols; i++) w[i] = float_to_fp16(fp16_to_float(w[i]) * wscale);
        }
        uint16_t* tw = (uint16_t*)malloc(sizeof(uint16_t) * rows * cols);
        transpose_block16((uint8_t*)w, (uint8_t*)tw, rows, cols, QUANT_FP16);
        t.data = createBuffer(s.dev.device, s.dev.physicalDevice, tw, sizeof(uint16_t) * rows * cols, MEMORY_VRAM);
        free(tw);
        free(w);
    } else if (q == QUANT_INT8) {
        QuantizedData qd = getDataINT8(seed, rows, cols);
        if (wscale != 1.0f) {
            for (int i = 0; i < rows * blocks; i++) {
                qd.scale[i] *= wscale;
                qd.z[i] *= wscale;
            }
        }
        uint8_t* tw = (uint8_t*)malloc(rows * cols);
        transpose_block16(qd.data, tw, rows, cols, QUANT_INT8);
        t.data = createBuffer(s.dev.device, s.dev.physicalDevice, tw, rows * cols, MEMORY_VRAM);
        t.scale = createBuffer(s.dev.device, s.dev.physicalDevice, qd.scale, sizeof(float) * rows * blocks, MEMORY_VRAM);
        t.zero = createBuffer(s.dev.device, s.dev.physicalDevice, qd.z, sizeof(float) * rows * blocks, MEMORY_VRAM);
        free(tw);
        free_quantized_data(qd);
    } else {
        QuantizedData qd = getDataINT4(seed, rows, cols);
        if (wscale != 1.0f) {
            for (int i = 0; i < rows * blocks; i++) {
                qd.scale[i] *= wscale;
                qd.z[i] *= wscale;
            }
        }
        uint8_t* tw = (uint8_t*)malloc(rows * cols / 2);
        transpose_block16(qd.data, tw, rows, cols, QUANT_INT4);
        t.data = createBuffer(s.dev.device, s.dev.physicalDevice, tw, rows * cols / 2, MEMORY_VRAM);
        t.scale = createBuffer(s.dev.device, s.dev.physicalDevice, qd.scale, sizeof(float) * rows * blocks, MEMORY_VRAM);
        t.zero = createBuffer(s.dev.device, s.dev.physicalDevice, qd.z, sizeof(float) * rows * blocks, MEMORY_VRAM);
        free(tw);
        free_quantized_data(qd);
    }

    return t;
}

static void destroyTensor(session s, tensor* t) {
    if (t->data.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->data);
    if (t->scale.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->scale);
    if (t->zero.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->zero);
}

model_weights createWeights(session s, const model_config* spec) {
    model_weights w = {0};

    float* theta = (float*)malloc(sizeof(float) * (MODEL_HEAD_DIM / 2));
    for (int i = 0; i < MODEL_HEAD_DIM / 2; i++) {
        theta[i] = (float)pow(1e6, -((double)i) / (MODEL_HEAD_DIM / 2));
    }
    w.theta = createBuffer(s.dev.device, s.dev.physicalDevice, theta, sizeof(float) * (MODEL_HEAD_DIM / 2), MEMORY_RAM);
    free(theta);

    float* gammaFinal = getData(77771, 1, MODEL_K);
    w.gammaFinal = createBuffer(s.dev.device, s.dev.physicalDevice, gammaFinal, sizeof(float) * MODEL_K, MEMORY_VRAM);
    free(gammaFinal);

    uint16_t* lm = getDataFP16(15001, MODEL_K, MODEL_VOCAB);
    uint16_t* tlm = (uint16_t*)malloc(sizeof(uint16_t) * MODEL_K * MODEL_VOCAB);
    transpose_block16((uint8_t*)lm, (uint8_t*)tlm, MODEL_K, MODEL_VOCAB, QUANT_FP16);
    w.lmHead = createBuffer(s.dev.device, s.dev.physicalDevice, tlm, sizeof(uint16_t) * MODEL_K * MODEL_VOCAB, MEMORY_VRAM);
    w.embed = w.lmHead;
    free(tlm);
    free(lm);

    for (int L = 0; L < spec->layerCount; L++) {
        const layer* ly = &spec->layers[L];
        QuantType q = ly->attn.q;
        QuantType f = ly->ffn.q;

        float* gIn = getData(80001 + L, 1, MODEL_K);
        float* gF = getData(81001 + L, 1, MODEL_K);
        w.gammaIn[L] = createBuffer(s.dev.device, s.dev.physicalDevice, gIn, sizeof(float) * MODEL_K, MEMORY_VRAM);
        w.gammaF[L] = createBuffer(s.dev.device, s.dev.physicalDevice, gF, sizeof(float) * MODEL_K, MEMORY_VRAM);
        free(gIn);
        free(gF);

        if (ly->attn.type == ATTENTION_FULL) {
            w.proj[L] = createTensor(s, 82001 + L * 10, MODEL_K, MODEL_QKV_N, q, 1.0f);
            w.out[L] = createTensor(s, 83001 + L * 10, MODEL_K, MODEL_K, q, 1.0f);
        } else if (ly->attn.type == ATTENTION_DELTA) {
            w.proj[L] = createTensor(s, 82001 + L * 10, MODEL_K, MODEL_PROJ_N, q, 1.0f / 64.0f);
            w.out[L] = createTensor(s, 83001 + L * 10, MODEL_K, MODEL_K, q, 1.0f);
        }

        if (ly->ffn.type == FFN_SWIGLU) {
            w.gate[L] = createTensor(s, 84001 + L * 10, MODEL_K, MODEL_FFN_N, f, 1.0f);
            w.up[L] = createTensor(s, 85001 + L * 10, MODEL_K, MODEL_FFN_N, f, 1.0f);
            w.down[L] = createTensor(s, 86001 + L * 10, MODEL_FFN_N, MODEL_K, f, 1.0f);
        }
    }

    return w;
}

void destroyWeights(session s, model_weights* w) {
    for (int L = 0; L < MODEL_LAYERS; L++) {
        destroyTensor(s, &w->proj[L]);
        destroyTensor(s, &w->out[L]);
        destroyTensor(s, &w->gate[L]);
        destroyTensor(s, &w->up[L]);
        destroyTensor(s, &w->down[L]);
        destroyBuffer(s.dev.device, w->gammaIn[L]);
        destroyBuffer(s.dev.device, w->gammaF[L]);
    }
    destroyBuffer(s.dev.device, w->theta);
    destroyBuffer(s.dev.device, w->gammaFinal);
    destroyBuffer(s.dev.device, w->lmHead);
}
