#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <direct.h>
#include "weights.h"
#include "safetensors.h"

static int64_t weightBytes = 0;

#define TENSOR_CACHE_MAX 256
#define TENSOR_FILE_MAGIC 0x54454E53
#define CACHE_DIR "weights"

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

static void fatal(const char* msg) {
    fprintf(stderr, "weight load: %s\n", msg);
    exit(1);
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
    fprintf(stderr, "%s[%d]: %lld bytes (%.2f MB) | Total: %lld bytes (%.2f MB)\n",
            name, layer, (long long)b.size, (double)b.size / (1024.0 * 1024.0),
            (long long)weightBytes, (double)weightBytes / (1024.0 * 1024.0));
}

static tensor createTensor(session s, const char* name, int layer, int rows, int cols, QuantType q, float wscale, const float* mat) {
    tensor t = {0};
    t.q = q;
    t.rows = rows;
    t.cols = cols;
    int blocks = (cols + 255) / 256;

    cachedTensor* ct = cacheFind(name, q);
    if (ct == NULL) {
        char path[160];
        snprintf(path, sizeof(path), CACHE_DIR "/%s_%s.bin", name, quantSuffix(q));
        ct = tensorLoadFile(path, name, q, rows, cols, blocks);
        if (ct == NULL) {
            ct = tensorBuild(path, name, q, rows, cols, wscale, mat);
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

static void lname(char* buf, size_t cap, int L, const char* sub) {
    snprintf(buf, cap, "model.language_model.layers.%d.%s", L, sub);
}

static const sa_tensor* require(const safetensors* sf, const char* name) {
    const sa_tensor* t = safetensors_find(sf, name);
    if (!t) {
        char msg[320];
        snprintf(msg, sizeof(msg), "missing tensor %s", name);
        fatal(msg);
    }
    return t;
}

static float* loadVec(const safetensors* sf, const char* name, int len) {
    const sa_tensor* t = require(sf, name);
    int64_t n = 0;
    float* v = safetensors_load_f32(sf, t, &n);
    if (!v || n != len) fatal("vector length mismatch");
    return v;
}

static buffer loadVecBuffer(session s, const safetensors* sf, const char* name, int len, const char* label, int layer) {
    float* v = loadVec(sf, name, len);
    buffer b = createBuffer(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * len, MEMORY_VRAM);
    countBuffer(label, layer, b);
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
        if (!src || n != (int64_t)sizes[i] * engineRows) fatal("matrix length mismatch");
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

static void loadEmbedLike(session s, const safetensors* sf, const char* hfName, const char* name, int V, buffer* out) {
    int K = MODEL_K;
    char path[160];
    snprintf(path, sizeof(path), CACHE_DIR "/%s_%d_FP16.bin", name, V);
    cachedTensor* ct = tensorLoadFile(path, name, QUANT_FP16, K, V, (V + 255) / 256);
    if (ct == NULL) {
        const sa_tensor* t = require(sf, hfName);
        if (t->ndim < 2 || t->shape[0] != V || t->shape[1] != K) fatal("embedding shape mismatch");
        uint16_t* raw = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)V * K);
        FILE* f = sf->files[t->fileIndex];
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
    }
    *out = createBuffer(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM);
    countBuffer(name, -1, *out);
}

static buffer loadConv(session s, const safetensors* sf, const char* name, int layer) {
    const sa_tensor* t = require(sf, name);
    int64_t n = 0;
    float* v = safetensors_load_f32(sf, t, &n);
    if (!v) fatal("conv read error");
    uint16_t* h = (uint16_t*)malloc(sizeof(uint16_t) * n);
    for (int64_t i = 0; i < n; i++) h[i] = float_to_fp16(v[i]);
    free(v);
    buffer b = createBuffer(s.dev.device, s.dev.physicalDevice, h, sizeof(uint16_t) * n, MEMORY_VRAM);
    countBuffer("conv", layer, b);
    free(h);
    return b;
}

model_weights createWeights(session s, const model_config* spec, const char* weightDir, const char* customHead, const char* customEmbed) {
    model_weights w = {0};
    weightBytes = 0;
    cacheClear();
    _mkdir(CACHE_DIR);

    char p1[320], p2[320], p3[320], p4[320];
    snprintf(p1, sizeof(p1), "%s/model.safetensors-00001-of-00004.safetensors", weightDir);
    snprintf(p2, sizeof(p2), "%s/model.safetensors-00002-of-00004.safetensors", weightDir);
    snprintf(p3, sizeof(p3), "%s/model.safetensors-00003-of-00004.safetensors", weightDir);
    snprintf(p4, sizeof(p4), "%s/model.safetensors-00004-of-00004.safetensors", weightDir);
    const char* shards[4] = {p1, p2, p3, p4};

    safetensors sf, sfEmb, sfLm;
    if (safetensors_open(&sf, shards, 4) != 0) fatal("shard open failed");

    int customVocab = (customHead != NULL && customEmbed != NULL);
    if (customVocab) {
        if (safetensors_open(&sfLm, &customHead, 1) != 0) fatal("custom lm head open failed");
        if (safetensors_open(&sfEmb, &customEmbed, 1) != 0) fatal("custom embedding open failed");
    }

    float* theta = (float*)malloc(sizeof(float) * (MODEL_HEAD_DIM / 2));
    for (int i = 0; i < MODEL_HEAD_DIM / 2; i++) {
        theta[i] = (float)pow(1e7, -((double)i) / (MODEL_HEAD_DIM / 2));
    }
    w.theta = createBuffer(s.dev.device, s.dev.physicalDevice, theta, sizeof(float) * (MODEL_HEAD_DIM / 2), MEMORY_VRAM);
    countBuffer("theta", -1, w.theta);
    free(theta);

    w.gammaFinal = loadVecBuffer(s, &sf, "model.language_model.norm.weight", MODEL_K, "gammaFinal", -1);

    const safetensors* sfHead = customVocab ? &sfLm : &sf;
    const safetensors* sfEmbUse = customVocab ? &sfEmb : &sf;
    const sa_tensor* headT = require(sfHead, "lm_head.weight");
    if (headT->ndim < 2) fatal("lm head not a matrix");
    int V = (int)headT->shape[0];
    w.vocab = V;

    loadEmbedLike(s, sfHead, "lm_head.weight", "lmHead", V, &w.lmHead);
    loadEmbedLike(s, sfEmbUse, "model.language_model.embed_tokens.weight", "embed", V, &w.embed);

    char n1[256], n2[256], n3[256], n4[256];

    for (int L = 0; L < spec->layerCount; L++) {
        const layer* ly = &spec->layers[L];
        QuantType q = ly->attn.q;
        QuantType f = ly->ffn.q;

        lname(n1, sizeof(n1), L, "input_layernorm.weight");
        w.gammaIn[L] = loadVecBuffer(s, &sf, n1, MODEL_K, "gammaIn", L);
        lname(n1, sizeof(n1), L, "post_attention_layernorm.weight");
        w.gammaF[L] = loadVecBuffer(s, &sf, n1, MODEL_K, "gammaF", L);

        char projName[64], outName[64];
        snprintf(projName, sizeof(projName), "proj_%d", L);
        snprintf(outName, sizeof(outName), "out_%d", L);

        if (ly->attn.type == ATTENTION_FULL) {
            lname(n1, sizeof(n1), L, "self_attn.q_norm.weight");
            w.qNorm[L] = loadVecBuffer(s, &sf, n1, MODEL_HEAD_DIM, "qNorm", L);
            lname(n1, sizeof(n1), L, "self_attn.k_norm.weight");
            w.kNorm[L] = loadVecBuffer(s, &sf, n1, MODEL_HEAD_DIM, "kNorm", L);

            lname(n1, sizeof(n1), L, "self_attn.q_proj.weight");
            lname(n2, sizeof(n2), L, "self_attn.k_proj.weight");
            lname(n3, sizeof(n3), L, "self_attn.v_proj.weight");
            const char* pn[3] = {n1, n2, n3};
            int cols = 0;
            float* mat = buildEngineMatrix(&sf, pn, 3, MODEL_K, &cols);
            if (cols != MODEL_QKV_N) fatal("qkv projection width mismatch");
            w.proj[L] = createTensor(s, projName, L, MODEL_K, MODEL_QKV_N, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "self_attn.o_proj.weight");
            const char* on[1] = {n1};
            mat = buildEngineMatrix(&sf, on, 1, MODEL_K, &cols);
            w.out[L] = createTensor(s, outName, L, MODEL_K, MODEL_K, q, 1.0f, mat);
            free(mat);
        } else {
            lname(n1, sizeof(n1), L, "linear_attn.conv1d.weight");
            w.conv[L] = loadConv(s, &sf, n1, L);
            lname(n1, sizeof(n1), L, "linear_attn.A_log");
            w.aLog[L] = loadVecBuffer(s, &sf, n1, MODEL_N_V, "aLog", L);
            lname(n1, sizeof(n1), L, "linear_attn.dt_bias");
            w.dtBias[L] = loadVecBuffer(s, &sf, n1, MODEL_N_V, "dtBias", L);
            lname(n1, sizeof(n1), L, "linear_attn.norm.weight");
            w.attnNorm[L] = loadVecBuffer(s, &sf, n1, MODEL_DIM, "attnNorm", L);

            lname(n1, sizeof(n1), L, "linear_attn.in_proj_qkv.weight");
            lname(n2, sizeof(n2), L, "linear_attn.in_proj_z.weight");
            lname(n3, sizeof(n3), L, "linear_attn.in_proj_a.weight");
            lname(n4, sizeof(n4), L, "linear_attn.in_proj_b.weight");
            const char* pn[4] = {n1, n2, n3, n4};
            int cols = 0;
            float* mat = buildEngineMatrix(&sf, pn, 4, MODEL_K, &cols);
            if (cols != MODEL_PROJ_N) fatal("delta projection width mismatch");
            w.proj[L] = createTensor(s, projName, L, MODEL_K, MODEL_PROJ_N, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "linear_attn.out_proj.weight");
            const char* on[1] = {n1};
            mat = buildEngineMatrix(&sf, on, 1, MODEL_K, &cols);
            w.out[L] = createTensor(s, outName, L, MODEL_K, MODEL_K, q, 1.0f, mat);
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
            float* mat = buildEngineMatrix(&sf, gn, 1, MODEL_K, &cols);
            if (cols != MODEL_FFN_N) fatal("gate width mismatch");
            w.gate[L] = createTensor(s, gateName, L, MODEL_K, MODEL_FFN_N, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.up_proj.weight");
            mat = buildEngineMatrix(&sf, gn, 1, MODEL_K, &cols);
            w.up[L] = createTensor(s, upName, L, MODEL_K, MODEL_FFN_N, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.down_proj.weight");
            mat = buildEngineMatrix(&sf, gn, 1, MODEL_FFN_N, &cols);
            if (cols != MODEL_K) fatal("down projection width mismatch");
            w.down[L] = createTensor(s, downName, L, MODEL_FFN_N, MODEL_K, f, 1.0f, mat);
            free(mat);
        }
    }

    safetensors_close(&sf);
    if (customVocab) {
        safetensors_close(&sfLm);
        safetensors_close(&sfEmb);
    }
    cacheClear();

    fprintf(stderr, "total weights: %lld bytes (%.2f MB, %.2f GB)\n",
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
        if (w->qNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->qNorm[L]);
        if (w->kNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->kNorm[L]);
        if (w->conv[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->conv[L]);
        if (w->aLog[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->aLog[L]);
        if (w->dtBias[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->dtBias[L]);
        if (w->attnNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->attnNorm[L]);
    }
    destroyBuffer(s.dev.device, w->theta);
    destroyBuffer(s.dev.device, w->gammaFinal);
    destroyBuffer(s.dev.device, w->lmHead);
}