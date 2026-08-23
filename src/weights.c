#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <direct.h>
#include "weights.h"

static int64_t weightBytes = 0;

#define TENSOR_CACHE_MAX 24
#define TENSOR_FILE_MAGIC 0x54454E53

typedef struct {
    char name[64];
    QuantType q;
    int rows;
    int cols;
    uint8_t* data;
    int dataBytes;
    float* scale;
    float* zero;
    int scaleCount;
} cachedTensor;

static cachedTensor tensorCache[TENSOR_CACHE_MAX];
static int tensorCacheCount = 0;

static const char* quantSuffix(QuantType q) {
    if (q == QUANT_FP16) return "FP16";
    if (q == QUANT_INT8) return "INT8";
    return "INT4";
}

static cachedTensor* cacheFind(const char* name, QuantType q) {
    for (int i = 0; i < tensorCacheCount; i++) {
        if (tensorCache[i].q == q && strcmp(tensorCache[i].name, name) == 0) {
            return &tensorCache[i];
        }
    }
    return NULL;
}

static cachedTensor* cacheStore(const char* name, QuantType q, int rows, int cols, uint8_t* data, int dataBytes, float* scale, float* zero, int scaleCount) {
    cachedTensor* ct = &tensorCache[tensorCacheCount++];
    memset(ct, 0, sizeof(cachedTensor));
    snprintf(ct->name, sizeof(ct->name), "%s", name);
    ct->q = q;
    ct->rows = rows;
    ct->cols = cols;
    ct->data = data;
    ct->dataBytes = dataBytes;
    ct->scale = scale;
    ct->zero = zero;
    ct->scaleCount = scaleCount;
    return ct;
}

static void cacheClear(void) {
    for (int i = 0; i < tensorCacheCount; i++) {
        free(tensorCache[i].data);
        free(tensorCache[i].scale);
        free(tensorCache[i].zero);
    }
    tensorCacheCount = 0;
}

static void tensorWriteFile(const char* path, QuantType q, int rows, int cols, const uint8_t* data, int dataBytes, const float* scale, const float* zero, int scaleCount) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int header[4] = {TENSOR_FILE_MAGIC, rows, cols, (int)q};
    fwrite(header, sizeof(int), 4, f);
    fwrite(data, 1, dataBytes, f);
    if (q != QUANT_FP16) {
        fwrite(scale, sizeof(float), scaleCount, f);
        fwrite(zero, sizeof(float), scaleCount, f);
    }
    fclose(f);
}

static cachedTensor* tensorLoadFile(const char* path, const char* name, QuantType q, int rows, int cols, int blocks) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int header[4];
    if (fread(header, sizeof(int), 4, f) != 4) { fclose(f); return NULL; }
    if (header[0] != TENSOR_FILE_MAGIC || header[1] != rows || header[2] != cols || header[3] != (int)q) { fclose(f); return NULL; }
    int dataBytes;
    if (q == QUANT_FP16) dataBytes = rows * cols * 2;
    else if (q == QUANT_INT8) dataBytes = rows * cols;
    else dataBytes = rows * cols / 2;
    int scaleCount = rows * blocks;
    uint8_t* data = (uint8_t*)malloc(dataBytes);
    float* scale = NULL;
    float* zero = NULL;
    if (fread(data, 1, dataBytes, f) != (size_t)dataBytes) { free(data); fclose(f); return NULL; }
    if (q != QUANT_FP16) {
        scale = (float*)malloc(sizeof(float) * scaleCount);
        zero = (float*)malloc(sizeof(float) * scaleCount);
        if (fread(scale, sizeof(float), scaleCount, f) != (size_t)scaleCount ||
            fread(zero, sizeof(float), scaleCount, f) != (size_t)scaleCount) {
            free(data); free(scale); free(zero); fclose(f); return NULL;
        }
    }
    fclose(f);
    return cacheStore(name, q, rows, cols, data, dataBytes, scale, zero, scaleCount);
}

static cachedTensor* tensorGenerate(const char* path, const char* name, QuantType q, int rows, int cols, int seed, float wscale) {
    int blocks = (cols + 255) / 256;
    int scaleCount = rows * blocks;
    cachedTensor* ct;
    if (q == QUANT_FP16) {
        uint16_t* w = getDataFP16(seed, rows, cols);
        if (wscale != 1.0f) {
            for (int i = 0; i < rows * cols; i++) w[i] = float_to_fp16(fp16_to_float(w[i]) * wscale);
        }
        uint16_t* tw = (uint16_t*)malloc(sizeof(uint16_t) * rows * cols);
        transpose_block16((uint8_t*)w, (uint8_t*)tw, rows, cols, QUANT_FP16);
        free(w);
        ct = cacheStore(name, q, rows, cols, (uint8_t*)tw, rows * cols * 2, NULL, NULL, 0);
        tensorWriteFile(path, q, rows, cols, ct->data, ct->dataBytes, NULL, NULL, 0);
    } else {
        QuantizedData qd = (q == QUANT_INT8) ? getDataINT8(seed, rows, cols) : getDataINT4(seed, rows, cols);
        if (wscale != 1.0f) {
            for (int i = 0; i < rows * blocks; i++) {
                qd.scale[i] *= wscale;
                qd.z[i] *= wscale;
            }
        }
        int dataBytes = (q == QUANT_INT8) ? rows * cols : rows * cols / 2;
        uint8_t* tw = (uint8_t*)malloc(dataBytes);
        transpose_block16(qd.data, tw, rows, cols, q);
        free(qd.data);
        ct = cacheStore(name, q, rows, cols, tw, dataBytes, qd.scale, qd.z, scaleCount);
        tensorWriteFile(path, q, rows, cols, ct->data, ct->dataBytes, ct->scale, ct->zero, ct->scaleCount);
    }
    return ct;
}

static void countBuffer(const char* name, int layer, buffer b) {
    weightBytes += b.size;
    printf("%s[%d]: %lld bytes (%.2f MB) | Total: %lld bytes (%.2f MB)\n", name, layer, (long long)b.size, (double)b.size / (1024.0 * 1024.0), (long long)weightBytes, (double)weightBytes / (1024.0 * 1024.0));
}

static tensor createTensor(session s, const char* name, int layer, int seed, int rows, int cols, QuantType q, float wscale) {
    tensor t = {0};
    t.q = q;
    t.rows = rows;
    t.cols = cols;
    int blocks = (cols + 255) / 256;

    cachedTensor* ct = cacheFind(name, q);
    if (ct == NULL) {
        char path[160];
        snprintf(path, sizeof(path), "demoWeight/%s_%s.bin", name, quantSuffix(q));
        ct = tensorLoadFile(path, name, q, rows, cols, blocks);
        if (ct == NULL) {
            ct = tensorGenerate(path, name, q, rows, cols, seed, wscale);
        }
    }

    t.data = createBuffer(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM);
    countBuffer(name, layer, t.data);
    if (q != QUANT_FP16) {
        char label[64];
        snprintf(label, sizeof(label), "%s-scale", name);
        t.scale = createBuffer(s.dev.device, s.dev.physicalDevice, ct->scale, sizeof(float) * ct->scaleCount, MEMORY_VRAM);
        countBuffer(label, layer, t.scale);
        snprintf(label, sizeof(label), "%s-zero", name);
        t.zero = createBuffer(s.dev.device, s.dev.physicalDevice, ct->zero, sizeof(float) * ct->scaleCount, MEMORY_VRAM);
        countBuffer(label, layer, t.zero);
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
    weightBytes = 0;
    cacheClear();
    _mkdir("demoWeight");

    float* theta = (float*)malloc(sizeof(float) * (MODEL_HEAD_DIM / 2));
    for (int i = 0; i < MODEL_HEAD_DIM / 2; i++) {
        theta[i] = (float)pow(1e6, -((double)i) / (MODEL_HEAD_DIM / 2));
    }
    w.theta = createBuffer(s.dev.device, s.dev.physicalDevice, theta, sizeof(float) * (MODEL_HEAD_DIM / 2), MEMORY_VRAM);
    countBuffer("theta", -1, w.theta);
    free(theta);

    float* gammaFinal = getData(77771, 1, MODEL_K);
    w.gammaFinal = createBuffer(s.dev.device, s.dev.physicalDevice, gammaFinal, sizeof(float) * MODEL_K, MEMORY_VRAM);
    countBuffer("gammaFinal", -1, w.gammaFinal);
    free(gammaFinal);

    uint16_t* lm = getDataFP16(15001, MODEL_K, MODEL_VOCAB);
    uint16_t* tlm = (uint16_t*)malloc(sizeof(uint16_t) * MODEL_K * MODEL_VOCAB);
    transpose_block16((uint8_t*)lm, (uint8_t*)tlm, MODEL_K, MODEL_VOCAB, QUANT_FP16);
    w.lmHead = createBuffer(s.dev.device, s.dev.physicalDevice, tlm, sizeof(uint16_t) * MODEL_K * MODEL_VOCAB, MEMORY_VRAM);
    countBuffer("lmHead", -1, w.lmHead);
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
        countBuffer("gammaIn", L, w.gammaIn[L]);
        w.gammaF[L] = createBuffer(s.dev.device, s.dev.physicalDevice, gF, sizeof(float) * MODEL_K, MEMORY_VRAM);
        countBuffer("gammaF", L, w.gammaF[L]);
        free(gIn);
        free(gF);

        if (ly->attn.type == ATTENTION_FULL) {
            w.proj[L] = createTensor(s, "proj", L, 82001 + L * 10, MODEL_K, MODEL_QKV_N, q, 1.0f);
            w.out[L] = createTensor(s, "out", L, 83001 + L * 10, MODEL_K, MODEL_K, q, 1.0f);
        } else if (ly->attn.type == ATTENTION_DELTA) {
            w.proj[L] = createTensor(s, "projDelta", L, 82001 + L * 10, MODEL_K, MODEL_PROJ_N, q, 1.0f / 64.0f);
            w.out[L] = createTensor(s, "outDelta", L, 83001 + L * 10, MODEL_K, MODEL_K, q, 1.0f);
        }

        if (ly->ffn.type == FFN_SWIGLU) {
            w.gate[L] = createTensor(s, "gate", L, 84001 + L * 10, MODEL_K, MODEL_FFN_N, f, 1.0f);
            w.up[L] = createTensor(s, "up", L, 85001 + L * 10, MODEL_K, MODEL_FFN_N, f, 1.0f);
            w.down[L] = createTensor(s, "down", L, 86001 + L * 10, MODEL_FFN_N, MODEL_K, f, 1.0f);
        }
    }

    cacheClear();

    printf("total weights: %lld bytes (%.2f MB, %.2f GB)\n",
           (long long)weightBytes,
           (double)weightBytes / (1024.0 * 1024.0),
           (double)weightBytes / (1024.0 * 1024.0 * 1024.0));

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