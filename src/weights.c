#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <direct.h>
#include <windows.h>
#include "weights.h"
#include "safetensors.h"

static int64_t weightBytes = 0;
static int verboseWeights = 0;

static const char* SPINNER = "-\\|/";
static int spinnerIdx = 0;

static void printProgress(void) {
    if (verboseWeights) return;
    fprintf(stderr, "\r[%c] loading weights: %.2f MB      ",
            SPINNER[spinnerIdx++ % 4],
            (double)weightBytes / (1024.0 * 1024.0));
}

#define TENSOR_CACHE_MAX 256
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
static char cacheDir[256] = "weights";

static const char* quantSuffix(QuantType q) {
    if (q == QUANT_FP16) return "FP16";
    if (q == QUANT_INT8) return "INT8";
    return "INT4";
}

static void fatal(const char* msg) {
    fprintf(stderr, "weight load: %s\n", msg);
    exit(1);
}

#define MAX_WEIGHT_BUFS 2560

static buffer g_wbufs[MAX_WEIGHT_BUFS];
static int g_wbufsCount = 0;

static void registerWeightBuffer(buffer b) {
    if (g_wbufsCount < MAX_WEIGHT_BUFS) {
        g_wbufs[g_wbufsCount++] = b;
    }
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
    if (tensorCacheCount >= TENSOR_CACHE_MAX) fatal("tensor cache overflow");
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

static void cacheRelease(cachedTensor* ct) {
    if (ct == NULL) return;
    free(ct->data);
    free(ct->scale);
    free(ct->zero);
    ct->data = NULL;
    ct->scale = NULL;
    ct->zero = NULL;
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
    if (fread(header, sizeof(int), 4, f) != 4) {
        fclose(f);
        return NULL;
    }
    if (header[0] != TENSOR_FILE_MAGIC || header[1] != rows || header[2] != cols || header[3] != (int)q) {
        fclose(f);
        return NULL;
    }
    int dataBytes;
    if (q == QUANT_FP16) dataBytes = rows * cols * 2;
    else if (q == QUANT_INT8) dataBytes = rows * cols;
    else dataBytes = rows * cols / 2;
    int scaleCount = rows * blocks;
    uint8_t* data = (uint8_t*)malloc(dataBytes);
    float* scale = NULL;
    float* zero = NULL;
    if (fread(data, 1, dataBytes, f) != (size_t)dataBytes) {
        free(data);
        fclose(f);
        return NULL;
    }
    if (q != QUANT_FP16) {
        scale = (float*)malloc(sizeof(float) * scaleCount);
        zero = (float*)malloc(sizeof(float) * scaleCount);
        if (fread(scale, sizeof(float), scaleCount, f) != (size_t)scaleCount ||
            fread(zero, sizeof(float), scaleCount, f) != (size_t)scaleCount) {
            free(data);
            free(scale);
            free(zero);
            fclose(f);
            return NULL;
        }
    }
    fclose(f);
    return cacheStore(name, q, rows, cols, data, dataBytes, scale, zero, scaleCount);
}

static cachedTensor* cacheGet(const char* name, QuantType q, int rows, int cols) {
    cachedTensor* ct = cacheFind(name, q);
    if (ct != NULL) return ct;
    int blocks = (cols + 255) / 256;
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, name, quantSuffix(q));
    return tensorLoadFile(path, name, q, rows, cols, blocks);
}

static cachedTensor* tensorBuild(const char* path, const char* name, QuantType q, int rows, int cols, float wscale, const float* mat) {
    int blocks = (cols + 255) / 256;
    int scaleCount = rows * blocks;
    int64_t total = (int64_t)rows * cols;
    cachedTensor* ct;

    if (q == QUANT_FP16) {
        uint16_t* w = (uint16_t*)malloc(sizeof(uint16_t) * total);
        for (int64_t i = 0; i < total; i++) w[i] = float_to_fp16(mat[i] * wscale);
        uint16_t* tw = (uint16_t*)malloc(sizeof(uint16_t) * total);
        transpose_block16((uint8_t*)w, (uint8_t*)tw, rows, cols, QUANT_FP16);
        free(w);
        ct = cacheStore(name, q, rows, cols, (uint8_t*)tw, (int)(total * 2), NULL, NULL, 0);
        tensorWriteFile(path, q, rows, cols, ct->data, ct->dataBytes, NULL, NULL, 0);
    } else {
        QuantizedData qd = (q == QUANT_INT8) ? quantizeDataINT8(mat, rows, cols) : quantizeDataINT4(mat, rows, cols);
        if (wscale != 1.0f) {
            for (int i = 0; i < scaleCount; i++) {
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
    if (verboseWeights) {
        fprintf(stderr, "%s[%d]: %lld bytes (%.2f MB) | Total: %lld bytes (%.2f MB)\n",
                name, layer, (long long)b.size, (double)b.size / (1024.0 * 1024.0),
                (long long)weightBytes, (double)weightBytes / (1024.0 * 1024.0));
    } else {
        printProgress();
    }
}

static tensor createTensor(session s, const char* name, int layer, int rows, int cols, QuantType q, float wscale, const float* mat) {
    tensor t = {0};
    t.q = q;
    t.rows = rows;
    t.cols = cols;

    cachedTensor* ct = cacheGet(name, q, rows, cols);
    if (ct == NULL) {
        char path[384];
        snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, name, quantSuffix(q));
        ct = tensorBuild(path, name, q, rows, cols, wscale, mat);
    }

    t.data = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM, name);
    countBuffer(name, layer, t.data);
    registerWeightBuffer(t.data);
    if (q != QUANT_FP16) {
        char label[80];
        snprintf(label, sizeof(label), "%s-scale", name);
        t.scale = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->scale, sizeof(float) * ct->scaleCount, MEMORY_VRAM, label);
        countBuffer(label, layer, t.scale);
        registerWeightBuffer(t.scale);
        snprintf(label, sizeof(label), "%s-zero", name);
        t.zero = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->zero, sizeof(float) * ct->scaleCount, MEMORY_VRAM, label);
        countBuffer(label, layer, t.zero);
        registerWeightBuffer(t.zero);
    }
    cacheRelease(ct);

    return t;
}

static void destroyTensor(session s, tensor* t) {
    if (t->data.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->data);
    if (t->scale.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->scale);
    if (t->zero.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->zero);
}

static void lname(char* buf, size_t cap, int L, const char* sub) {
    snprintf(buf, cap, "model.language_model.layers.%d.%s", L, sub);
}

static const sa_tensor* require(const safetensors* sf, const char* name) {
    const sa_tensor* t = sf ? safetensors_find(sf, name) : NULL;
    if (!t) {
        char msg[320];
        if (sf == NULL) {
            snprintf(msg, sizeof(msg), "cache miss for %s and safetensors not available", name);
        } else {
            snprintf(msg, sizeof(msg), "missing tensor %s", name);
        }
        fatal(msg);
    }
    return t;
}

#define VEC_FILE_MAGIC 0x56454353

static float* loadVecRaw(const char* cacheName, int len) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int header[2];
    if (fread(header, sizeof(int), 2, f) != 2 || header[0] != VEC_FILE_MAGIC || header[1] != len) {
        fclose(f);
        return NULL;
    }
    float* v = (float*)malloc(sizeof(float) * len);
    if (fread(v, sizeof(float), len, f) != (size_t)len) {
        free(v);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return v;
}

static float* loadVecRawAny(const char* cacheName, int64_t* outLen) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int header[2];
    if (fread(header, sizeof(int), 2, f) != 2 || header[0] != VEC_FILE_MAGIC || header[1] <= 0) {
        fclose(f);
        return NULL;
    }
    int len = header[1];
    float* v = (float*)malloc(sizeof(float) * len);
    if (fread(v, sizeof(float), len, f) != (size_t)len) {
        free(v);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *outLen = len;
    return v;
}

static void saveVecRaw(const char* cacheName, const float* v, int len) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int header[2] = {VEC_FILE_MAGIC, len};
    fwrite(header, sizeof(int), 2, f);
    fwrite(v, sizeof(float), len, f);
    fclose(f);
}

static buffer loadVecBuffer(session s, const safetensors* sf, const char* hfName, int len, const char* label, int layer, int addOne) {
    char cacheName[80];
    snprintf(cacheName, sizeof(cacheName), "vec_%s_%d", label, layer);
    float* v = loadVecRaw(cacheName, len);
    if (v == NULL) {
        const sa_tensor* t = require(sf, hfName);
        int64_t n = 0;
        v = safetensors_load_f32(sf, t, &n);
        if (!v || n != len) fatal("vector length mismatch");
        if (addOne) {
            for (int i = 0; i < len; i++) v[i] += 1.0f;
        }
        saveVecRaw(cacheName, v, len);
    }
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * len, MEMORY_VRAM, label);
    countBuffer(label, layer, b);
    registerWeightBuffer(b);
    free(v);
    return b;
}

static float* buildEngineMatrix(const safetensors* sf, const char** hfNames, int hfCount, int engineRows, int* outCols) {
    int total = 0;
    int sizes[32];
    if (hfCount > 32) fatal("too many concat sources");
    for (int i = 0; i < hfCount; i++) {
        const sa_tensor* t = require(sf, hfNames[i]);
        if (t->ndim < 2) fatal("matrix tensor expected");
        sizes[i] = (int)t->shape[0];
        total += sizes[i];
    }
    float* hf = (float*)malloc(sizeof(float) * (size_t)total * engineRows);
    int off = 0;
    for (int i = 0; i < hfCount; i++) {
        const sa_tensor* t = safetensors_find(sf, hfNames[i]);
        int64_t n = 0;
        float* src = safetensors_load_f32(sf, t, &n);
        if (!src || n != (int64_t)sizes[i] * engineRows) {
            char buf[256];
            snprintf(buf, sizeof(buf), "matrix length mismatch: %s got=%lld want=%lld", hfNames[i], (long long)n, (long long)sizes[i] * engineRows);
            fatal(buf);
        }
        memcpy(hf + (size_t)off * engineRows, src, sizeof(float) * n);
        free(src);
        off += sizes[i];
    }
    float* eng = (float*)malloc(sizeof(float) * (size_t)total * engineRows);
    transpose(hf, eng, total, engineRows);
    free(hf);
    *outCols = total;
    return eng;
}

static float* buildQkvMatrix(const safetensors* sf, const char* qn, const char* kn, const char* vn, int engineRows, int headDim, int heads, int* outCols) {
    const sa_tensor* tq = require(sf, qn);
    const sa_tensor* tk = require(sf, kn);
    const sa_tensor* tv = require(sf, vn);
    int qRows = (int)tq->shape[0];
    int kRows = (int)tk->shape[0];
    int vRows = (int)tv->shape[0];
    int total = qRows + kRows + vRows;
    int hd = headDim;
    int qPart = heads * hd;

    float* hf = (float*)malloc(sizeof(float) * (size_t)total * engineRows);

    int64_t n = 0;
    float* src = safetensors_load_f32(sf, tq, &n);
    if (!src || n != (int64_t)qRows * engineRows) fatal("q proj length mismatch");
    for (int c = 0; c < qRows; c++) {
        int head, dim, srcRow;
        if (c < qPart) {
            head = c / hd;
            dim = c % hd;
            srcRow = head * (2 * hd) + dim;
        } else {
            head = (c - qPart) / hd;
            dim = (c - qPart) % hd;
            srcRow = head * (2 * hd) + hd + dim;
        }
        memcpy(hf + (size_t)c * engineRows, src + (size_t)srcRow * engineRows, sizeof(float) * engineRows);
    }
    free(src);

    float* sk = safetensors_load_f32(sf, tk, &n);
    if (!sk || n != (int64_t)kRows * engineRows) fatal("k proj length mismatch");
    memcpy(hf + (size_t)qRows * engineRows, sk, sizeof(float) * n);
    free(sk);

    float* sv = safetensors_load_f32(sf, tv, &n);
    if (!sv || n != (int64_t)vRows * engineRows) fatal("v proj length mismatch");
    memcpy(hf + (size_t)(qRows + kRows) * engineRows, sv, sizeof(float) * n);
    free(sv);

    float* eng = (float*)malloc(sizeof(float) * (size_t)total * engineRows);
    transpose(hf, eng, total, engineRows);
    free(hf);
    *outCols = total;
    return eng;
}

static void loadEmbedLike(session s, const safetensors* sfFallback, const char* const* candPaths, int candCount, const char* hfName, const char* name, int V, int K, buffer* out) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%d_FP16.bin", cacheDir, name, V);
    cachedTensor* ct = tensorLoadFile(path, name, QUANT_FP16, K, V, (V + 255) / 256);
    if (ct == NULL) {
        const sa_tensor* t = NULL;
        safetensors sfCand;
        int opened = 0;
        for (int i = 0; i < candCount && t == NULL; i++) {
            if (candPaths[i] == NULL) continue;
            const char* p = candPaths[i];
            if (safetensors_open(&sfCand, &p, 1) == 0) {
                t = safetensors_find(&sfCand, hfName);
                if (t != NULL) {
                    opened = 1;
                } else {
                    safetensors_close(&sfCand);
                }
            }
        }
        const safetensors* sfUse = opened ? &sfCand : sfFallback;
        if (t == NULL && sfUse != NULL) t = safetensors_find(sfUse, hfName);
        if (t == NULL) t = require(sfUse, hfName);
        if (t->ndim < 2 || t->shape[0] != V || t->shape[1] != K) fatal("embedding shape mismatch");
        uint16_t* raw = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)V * K);
        FILE* f = sfUse->files[t->fileIndex];
        _fseeki64(f, t->offset, SEEK_SET);
        if (fread(raw, sizeof(uint16_t), (size_t)V * K, f) != (size_t)V * K) fatal("embedding read error");
        uint16_t* eng = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)K * V);
        for (int v = 0; v < V; v++) {
            for (int k = 0; k < K; k++) {
                eng[(size_t)k * V + v] = float_to_fp16(bf16_to_float(raw[(size_t)v * K + k]));
            }
        }
        free(raw);
        uint16_t* tw = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)K * V);
        transpose_block16((uint8_t*)eng, (uint8_t*)tw, K, V, QUANT_FP16);
        free(eng);
        ct = cacheStore(name, QUANT_FP16, K, V, (uint8_t*)tw, K * V * 2, NULL, NULL, 0);
        tensorWriteFile(path, QUANT_FP16, K, V, ct->data, ct->dataBytes, NULL, NULL, 0);
        if (opened) safetensors_close(&sfCand);
    }
    *out = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM, name);
    countBuffer(name, -1, *out);
    registerWeightBuffer(*out);
    cacheRelease(ct);
}

static buffer loadConv(session s, const safetensors* sf, const char* name, int layer) {
    char cacheName[80];
    snprintf(cacheName, sizeof(cacheName), "conv_%d", layer);
    const sa_tensor* t = sf ? safetensors_find(sf, name) : NULL;
    float* v;
    int64_t n = 0;
    if (t != NULL) {
        v = safetensors_load_f32(sf, t, &n);
        if (!v) fatal("conv read error");
        saveVecRaw(cacheName, v, (int)n);
    } else {
        v = loadVecRawAny(cacheName, &n);
        if (v == NULL) {
            char msg[320];
            snprintf(msg, sizeof(msg), "cache miss for %s and safetensors not available", name);
            fatal(msg);
        }
    }
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * n, MEMORY_VRAM, cacheName);
    countBuffer("conv", layer, b);
    registerWeightBuffer(b);
    free(v);
    return b;
}

static int findShards(const char* dir, char out[][512], int max) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/model.safetensors*.safetensors", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    int count = 0;
    do {
        if (count >= max) break;
        snprintf(out[count], 512, "%s/%s", dir, fd.cFileName);
        count++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);

    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (strcmp(out[i], out[j]) > 0) {
                char tmp[512];
                strcpy(tmp, out[i]);
                strcpy(out[i], out[j]);
                strcpy(out[j], tmp);
            }
        }
    }
    return count;
}

static int findVocabFile(const char* dir, const char* prefix, char* out, size_t cap) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/vocab/%s.*.safetensors", dir, prefix);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    snprintf(out, cap, "%s/vocab/%s", dir, fd.cFileName);
    FindClose(h);
    return 1;
}

static int cacheFileExists(const char* name, QuantType q, int rows, int cols) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, name, quantSuffix(q));
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int header[4];
    int ok = (fread(header, sizeof(int), 4, f) == 4 &&
              header[0] == TENSOR_FILE_MAGIC && header[1] == rows && header[2] == cols && header[3] == (int)q);
    fclose(f);
    return ok;
}

static int vecCacheExists(const char* cacheName, int len) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int header[2];
    int ok = (fread(header, sizeof(int), 2, f) == 2 &&
              header[0] == VEC_FILE_MAGIC && header[1] == len);
    fclose(f);
    return ok;
}

static int cacheComplete(const model_config* spec) {
    const model_dims* d = &spec->dims;
    char name[80];
    if (!vecCacheExists("vec_gammaFinal_-1", d->K)) return 0;
    if (!cacheFileExists("embed", QUANT_FP16, d->K, d->vocab)) return 0;
    if (!d->tied && !cacheFileExists("lmHead", QUANT_FP16, d->K, d->vocab)) return 0;
    for (int L = 0; L < d->layerCount; L++) {
        const layer* ly = &spec->layers[L];
        QuantType q = ly->attn.q;
        QuantType f = ly->ffn.q;
        char vecName[80];
        snprintf(vecName, sizeof(vecName), "vec_gammaIn_%d", L);
        if (!vecCacheExists(vecName, d->K)) return 0;
        snprintf(vecName, sizeof(vecName), "vec_gammaF_%d", L);
        if (!vecCacheExists(vecName, d->K)) return 0;
        if (ly->attn.type == ATTENTION_FULL) {
            snprintf(name, sizeof(name), "proj_%d", L);
            if (!cacheFileExists(name, q, d->K, d->qkvN)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_qNorm_%d", L);
            if (!vecCacheExists(vecName, d->headDim)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_kNorm_%d", L);
            if (!vecCacheExists(vecName, d->headDim)) return 0;
        } else {
            snprintf(name, sizeof(name), "proj_%d", L);
            if (!cacheFileExists(name, q, d->K, d->projN)) return 0;
            snprintf(vecName, sizeof(vecName), "conv_%d", L);
            if (!vecCacheExists(vecName, 4 * d->zqkvN)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_aLog_%d", L);
            if (!vecCacheExists(vecName, d->nV)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_dtBias_%d", L);
            if (!vecCacheExists(vecName, d->nV)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_attnNorm_%d", L);
            if (!vecCacheExists(vecName, d->dim)) return 0;
        }
        snprintf(name, sizeof(name), "out_%d", L);
        if (!cacheFileExists(name, q, d->K, d->K)) return 0;
        if (ly->ffn.type == FFN_SWIGLU) {
            snprintf(name, sizeof(name), "gate_%d", L);
            if (!cacheFileExists(name, f, d->K, d->ffnN)) return 0;
            snprintf(name, sizeof(name), "up_%d", L);
            if (!cacheFileExists(name, f, d->K, d->ffnN)) return 0;
            snprintf(name, sizeof(name), "down_%d", L);
            if (!cacheFileExists(name, f, d->ffnN, d->K)) return 0;
        }
    }
    return 1;
}

model_weights createWeights(session s, const model_config* spec, const char* weightDir, int verbose) {
    model_weights w = {0};
    const model_dims* d = &spec->dims;
    weightBytes = 0;
    verboseWeights = verbose;
    g_wbufsCount = 0;
    cacheClear();
    snprintf(cacheDir, sizeof(cacheDir), "weights/%s", spec->name);
    _mkdir("weights");
    _mkdir(cacheDir);

    char shardPaths[16][512];
    const char* shardPtrs[16];
    int shardCount = findShards(weightDir, shardPaths, 16);
    for (int i = 0; i < shardCount; i++) shardPtrs[i] = shardPaths[i];

    safetensors sf;
    int hasShards = 0;
    if (shardCount > 0 && safetensors_open(&sf, shardPtrs, shardCount) == 0) {
        hasShards = 1;
    }

    char headPath[512];
    char embedPath[512];
    int hasHead = findVocabFile(weightDir, "lm_head", headPath, sizeof(headPath));
    int hasEmbed = findVocabFile(weightDir, "embed_tokens", embedPath, sizeof(embedPath));
    if (d->tied) hasHead = 0;
    if (d->tied && !hasEmbed) {
        hasEmbed = findVocabFile(weightDir, "lm_head", embedPath, sizeof(embedPath));
    }

    const safetensors* sfMain = hasShards ? &sf : NULL;
    if (!hasShards && !hasHead && !hasEmbed && !cacheComplete(spec)) {
        fatal("no safetensors found and weight cache is incomplete");
    }

    int rotaryHalf = d->rotaryDim / 2;
    float* theta = (float*)malloc(sizeof(float) * rotaryHalf);
    for (int i = 0; i < rotaryHalf; i++) {
        theta[i] = (float)pow(d->ropeTheta, -((double)i) / rotaryHalf);
    }
    w.theta = createBufferNamed(s.dev.device, s.dev.physicalDevice, theta, sizeof(float) * rotaryHalf, MEMORY_VRAM, "theta");
    countBuffer("theta", -1, w.theta);
    registerWeightBuffer(w.theta);
    free(theta);

    w.gammaFinal = loadVecBuffer(s, sfMain, "model.language_model.norm.weight", d->K, "gammaFinal", -1, 1);

    int V = d->vocab;
    w.vocab = V;
    w.layerCount = d->layerCount;

    w.layerBufs = (buffer*)calloc((size_t)d->layerCount * 13, sizeof(buffer));
    w.gammaIn = w.layerBufs + 0 * d->layerCount;
    w.gammaF = w.layerBufs + 1 * d->layerCount;
    w.qNorm = w.layerBufs + 2 * d->layerCount;
    w.kNorm = w.layerBufs + 3 * d->layerCount;
    w.conv = w.layerBufs + 4 * d->layerCount;
    w.aLog = w.layerBufs + 5 * d->layerCount;
    w.dtBias = w.layerBufs + 6 * d->layerCount;
    w.attnNorm = w.layerBufs + 7 * d->layerCount;
    w.tensorBufs = (tensor*)calloc((size_t)d->layerCount * 5, sizeof(tensor));
    w.proj = w.tensorBufs + 0 * d->layerCount;
    w.out = w.tensorBufs + 1 * d->layerCount;
    w.gate = w.tensorBufs + 2 * d->layerCount;
    w.up = w.tensorBufs + 3 * d->layerCount;
    w.down = w.tensorBufs + 4 * d->layerCount;

    const char* embedCands[2];
    int embedCandCount = 0;
    if (hasEmbed) embedCands[embedCandCount++] = embedPath;

    const char* headCands[2];
    int headCandCount = 0;
    if (hasHead) headCands[headCandCount++] = headPath;
    if (hasEmbed) headCands[headCandCount++] = embedPath;

    loadEmbedLike(s, sfMain, embedCands, embedCandCount, "model.language_model.embed_tokens.weight", "embed", V, d->K, &w.embed);
    if (d->tied) {
        w.lmHead = w.embed;
    } else {
        loadEmbedLike(s, sfMain, headCands, headCandCount, "lm_head.weight", "lmHead", V, d->K, &w.lmHead);
    }

    char n1[256], n2[256], n3[256], n4[256];

    for (int L = 0; L < spec->dims.layerCount; L++) {
        const layer* ly = &spec->layers[L];
        QuantType q = ly->attn.q;
        QuantType f = ly->ffn.q;

        lname(n1, sizeof(n1), L, "input_layernorm.weight");
        w.gammaIn[L] = loadVecBuffer(s, sfMain, n1, d->K, "gammaIn", L, 1);
        lname(n1, sizeof(n1), L, "post_attention_layernorm.weight");
        w.gammaF[L] = loadVecBuffer(s, sfMain, n1, d->K, "gammaF", L, 1);

        char projName[64], outName[64];
        snprintf(projName, sizeof(projName), "proj_%d", L);
        snprintf(outName, sizeof(outName), "out_%d", L);

        if (ly->attn.type == ATTENTION_FULL) {
            lname(n1, sizeof(n1), L, "self_attn.q_norm.weight");
            w.qNorm[L] = loadVecBuffer(s, sfMain, n1, d->headDim, "qNorm", L, 1);
            lname(n1, sizeof(n1), L, "self_attn.k_norm.weight");
            w.kNorm[L] = loadVecBuffer(s, sfMain, n1, d->headDim, "kNorm", L, 1);

            lname(n1, sizeof(n1), L, "self_attn.q_proj.weight");
            lname(n2, sizeof(n2), L, "self_attn.k_proj.weight");
            lname(n3, sizeof(n3), L, "self_attn.v_proj.weight");
            int cols = 0;
            float* mat = NULL;
            if (cacheGet(projName, q, d->K, d->qkvN) == NULL) {
                mat = buildQkvMatrix(sfMain, n1, n2, n3, d->K, d->headDim, d->heads, &cols);
                if (cols != d->qkvN) fatal("qkv projection width mismatch");
            }
            w.proj[L] = createTensor(s, projName, L, d->K, d->qkvN, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "self_attn.o_proj.weight");
            const char* on[1] = {n1};
            mat = NULL;
            if (cacheGet(outName, q, d->K, d->K) == NULL) {
                mat = buildEngineMatrix(sfMain, on, 1, d->K, &cols);
            }
            w.out[L] = createTensor(s, outName, L, d->K, d->K, q, 1.0f, mat);
            free(mat);
        } else {
            lname(n1, sizeof(n1), L, "linear_attn.conv1d.weight");
            w.conv[L] = loadConv(s, sfMain, n1, L);
            lname(n1, sizeof(n1), L, "linear_attn.A_log");
            w.aLog[L] = loadVecBuffer(s, sfMain, n1, d->nV, "aLog", L, 0);
            lname(n1, sizeof(n1), L, "linear_attn.dt_bias");
            w.dtBias[L] = loadVecBuffer(s, sfMain, n1, d->nV, "dtBias", L, 0);
            lname(n1, sizeof(n1), L, "linear_attn.norm.weight");
            w.attnNorm[L] = loadVecBuffer(s, sfMain, n1, d->dim, "attnNorm", L, 0);

            lname(n1, sizeof(n1), L, "linear_attn.in_proj_qkv.weight");
            lname(n2, sizeof(n2), L, "linear_attn.in_proj_z.weight");
            lname(n3, sizeof(n3), L, "linear_attn.in_proj_a.weight");
            lname(n4, sizeof(n4), L, "linear_attn.in_proj_b.weight");
            const char* pn[4] = {n1, n2, n3, n4};
            int cols = 0;
            float* mat = NULL;
            if (cacheGet(projName, q, d->K, d->projN) == NULL) {
                mat = buildEngineMatrix(sfMain, pn, 4, d->K, &cols);
                if (cols != d->projN) fatal("delta projection width mismatch");
            }
            w.proj[L] = createTensor(s, projName, L, d->K, d->projN, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "linear_attn.out_proj.weight");
            const char* on[1] = {n1};
            mat = NULL;
            if (cacheGet(outName, q, d->K, d->K) == NULL) {
                mat = buildEngineMatrix(sfMain, on, 1, d->K, &cols);
            }
            w.out[L] = createTensor(s, outName, L, d->K, d->K, q, 1.0f, mat);
            free(mat);
        }

        if (ly->ffn.type == FFN_SWIGLU) {
            char gateName[64], upName[64], downName[64];
            snprintf(gateName, sizeof(gateName), "gate_%d", L);
            snprintf(upName, sizeof(upName), "up_%d", L);
            snprintf(downName, sizeof(downName), "down_%d", L);

            const char* gn[1] = {n1};
            int cols = 0;
            lname(n1, sizeof(n1), L, "mlp.gate_proj.weight");
            float* mat = NULL;
            if (cacheGet(gateName, f, d->K, d->ffnN) == NULL) {
                mat = buildEngineMatrix(sfMain, gn, 1, d->K, &cols);
                if (cols != d->ffnN) fatal("gate width mismatch");
            }
            w.gate[L] = createTensor(s, gateName, L, d->K, d->ffnN, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.up_proj.weight");
            mat = NULL;
            if (cacheGet(upName, f, d->K, d->ffnN) == NULL) {
                mat = buildEngineMatrix(sfMain, gn, 1, d->K, &cols);
            }
            w.up[L] = createTensor(s, upName, L, d->K, d->ffnN, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.down_proj.weight");
            mat = NULL;
            if (cacheGet(downName, f, d->ffnN, d->K) == NULL) {
                mat = buildEngineMatrix(sfMain, gn, 1, d->ffnN, &cols);
                if (cols != d->K) fatal("down projection width mismatch");
            }
            w.down[L] = createTensor(s, downName, L, d->ffnN, d->K, f, 1.0f, mat);
            free(mat);
        }
    }

    if (hasShards) safetensors_close(&sf);
    cacheClear();

    createTransferAndCopy(s.dev.device, s.dev.queue, g_wbufs, g_wbufsCount);

    if (!verboseWeights) {
        fprintf(stderr, "\r[OK] loaded weights: %.2f MB             \n",
                (double)weightBytes / (1024.0 * 1024.0));
    }
    fprintf(stderr, "total weights: %lld bytes (%.2f MB, %.2f GB)\n",
            (long long)weightBytes,
            (double)weightBytes / (1024.0 * 1024.0),
            (double)weightBytes / (1024.0 * 1024.0 * 1024.0));

    return w;
}

void destroyWeights(session s, model_weights* w) {
    for (int L = 0; L < w->layerCount; L++) {
        destroyTensor(s, &w->proj[L]);
        destroyTensor(s, &w->out[L]);
        destroyTensor(s, &w->gate[L]);
        destroyTensor(s, &w->up[L]);
        destroyTensor(s, &w->down[L]);
        destroyBuffer(s.dev.device, w->gammaIn[L]);
        destroyBuffer(s.dev.device, w->gammaF[L]);
        if (w->qNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->qNorm[L]);
        if (w->kNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->kNorm[L]);
        if (w->conv[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->conv[L]);
        if (w->aLog[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->aLog[L]);
        if (w->dtBias[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->dtBias[L]);
        if (w->attnNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->attnNorm[L]);
    }
    free(w->layerBufs);
    free(w->tensorBufs);
    destroyBuffer(s.dev.device, w->theta);
    destroyBuffer(s.dev.device, w->gammaFinal);
    destroyBuffer(s.dev.device, w->lmHead);
    if (w->lmHead.buffer != w->embed.buffer) {
        destroyBuffer(s.dev.device, w->embed);
    }
}

