#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <io.h>
#include <direct.h>
#include <windows.h>
#include "prune.h"
#include "safetensors.h"

#define EMBED_NAME "model.language_model.embed_tokens.weight"
#define GATHER_CHUNK 8192

static void pfatal(const char* msg) {
    fprintf(stderr, "prune: %s\n", msg);
    exit(1);
}

static void perr(const char* msg, const char* arg) {
    fprintf(stderr, "prune: %s: %s\n", msg, arg);
    exit(1);
}

static char* readAll(const char* path, long* outLen) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[len] = '\0';
    *outLen = len;
    return buf;
}

static int32_t* loadMapping(const char* path, int expected) {
    FILE* f = fopen(path, "rb");
    if (!f) perr("cannot open mapping", path);
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) pfatal("bad npy magic");
    unsigned char ver[2];
    if (fread(ver, 1, 2, f) != 2) pfatal("bad npy version");
    if (ver[0] != 1) pfatal("unsupported npy version");
    unsigned char hlen[2];
    if (fread(hlen, 1, 2, f) != 2) pfatal("bad npy header");
    int hl = hlen[0] | (hlen[1] << 8);
    char* hdr = (char*)malloc(hl + 1);
    if (!hdr || fread(hdr, 1, hl, f) != (size_t)hl) pfatal("bad npy header");
    hdr[hl] = '\0';
    if (!strstr(hdr, "'<i4'")) pfatal("mapping npy must be int32");
    int count = 0;
    char* shp = strstr(hdr, "'shape':");
    if (!shp || sscanf(shp, "'shape': (%d,", &count) != 1) pfatal("bad npy shape");
    free(hdr);
    if (count != expected) pfatal("mapping length mismatch vs vocab_size");
    int32_t* m = (int32_t*)malloc(sizeof(int32_t) * count);
    if (!m || fread(m, sizeof(int32_t), count, f) != (size_t)count) pfatal("mapping read error");
    fclose(f);
    return m;
}

static int copyFileTo(const char* src, const char* dstDir) {
    const char* base = strrchr(src, '/');
    const char* name = base ? base + 1 : src;
    char dst[512];
    snprintf(dst, sizeof(dst), "%s/%s", dstDir, name);
    if (fopen(dst, "rb") != NULL) return 0;
    BOOL ok = CopyFileA(src, dst, FALSE);
    if (!ok) perr("copy failed", src);
    return 1;
}

static void writeSafetensorsHeader(FILE* f, const char* name, const char* dtype, int rows, int cols, long dataBytes) {
    char entry[512];
    snprintf(entry, sizeof(entry),
             "{\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[0,%ld]},\"__metadata__\":{\"format\":\"pt\"}}",
             name, dtype, rows, cols, dataBytes);
    int raw = (int)strlen(entry);
    int pad = (-raw) & 7;
    uint64_t total = (uint64_t)(raw + pad);
    fwrite(&total, sizeof(uint64_t), 1, f);
    fwrite(entry, 1, raw, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
}

static void gatherRows(FILE* src, FILE* dst, int64_t dataStart, const int32_t* mapping, int V, int K) {
    int rowBytes = K * 2;
    uint16_t* chunk = (uint16_t*)malloc((size_t)rowBytes * GATHER_CHUNK);
    if (!chunk) pfatal("out of memory");
    for (int start = 0; start < V; start += GATHER_CHUNK) {
        int stop = start + GATHER_CHUNK;
        if (stop > V) stop = V;
        for (int v = start; v < stop; v++) {
            _fseeki64(src, dataStart + (int64_t)mapping[v] * rowBytes, SEEK_SET);
            if (fread(chunk + (size_t)(v - start) * K, rowBytes, 1, src) != 1) pfatal("shard read error");
        }
        fwrite(chunk, rowBytes, stop - start, dst);
    }
    free(chunk);
}

int pruneVocab(const char* modelDir, const char* sourceDir) {
    char vocabDir[512];
    snprintf(vocabDir, sizeof(vocabDir), "%s/vocab", modelDir);

    char probe[512];
    snprintf(probe, sizeof(probe), "%s/embed_tokens.*.safetensors", vocabDir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(probe, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        FindClose(h);
        return 0;
    }

    char path[512];
    long len = 0;

    snprintf(path, sizeof(path), "%s/config.json", modelDir);
    char* cfgRaw = readAll(path, &len);
    if (!cfgRaw) perr("cannot read", path);
    const char* hs = strstr(cfgRaw, "\"hidden_size\"");
    int K = 0;
    if (!hs || sscanf(hs, "\"hidden_size\"%*[^0-9]%d", &K) != 1 || K <= 0) pfatal("bad hidden_size");
    free(cfgRaw);

    snprintf(path, sizeof(path), "%s/quant_config.json", modelDir);
    char* qcRaw = readAll(path, &len);
    if (!qcRaw) perr("cannot read", path);
    const char* vs = strstr(qcRaw, "\"vocab_size\"");
    int V = 0;
    if (!vs || sscanf(vs, "\"vocab_size\"%*[^0-9]%d", &V) != 1 || V <= 0) pfatal("bad vocab_size");
    free(qcRaw);

    snprintf(path, sizeof(path), "%s/mapping.npy", sourceDir);
    int32_t* mapping = loadMapping(path, V);

    snprintf(path, sizeof(path), "%s/model.safetensors*.safetensors", modelDir);
    HANDLE hs2 = FindFirstFileA(path, &fd);
    if (hs2 == INVALID_HANDLE_VALUE) pfatal("no model shards found");
    char shardPath[512];
    snprintf(shardPath, sizeof(shardPath), "%s/%s", modelDir, fd.cFileName);
    FindClose(hs2);

    safetensors sf;
    const char* shardPtr = shardPath;
    if (safetensors_open(&sf, &shardPtr, 1) != 0) perr("cannot open shard", shardPath);
    const sa_tensor* t = safetensors_find(&sf, EMBED_NAME);
    if (!t) perr("missing tensor in shard", EMBED_NAME);
    int srcRows = (int)t->shape[0];
    int srcCols = (int)t->shape[1];
    if (srcCols != K) pfatal("embedding width mismatch vs config");
    for (int i = 0; i < V; i++) {
        if (mapping[i] < 0 || mapping[i] >= srcRows) pfatal("mapping id out of range");
    }
    FILE* src = sf.files[t->fileIndex];

    _mkdir(vocabDir);

    char dstPath[512];
    snprintf(dstPath, sizeof(dstPath), "%s/embed_tokens.%d.safetensors", vocabDir, V);
    FILE* dst = fopen(dstPath, "wb");
    if (!dst) perr("cannot create", dstPath);
    long dataBytes = (long)V * K * 2;
    writeSafetensorsHeader(dst, EMBED_NAME, "BF16", V, K, dataBytes);
    gatherRows(src, dst, t->offset, mapping, V, K);
    fclose(dst);
    safetensors_close(&sf);
    free(mapping);

    const char* tokFiles[] = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "mapping.npy"};
    int copied = 0;
    for (int i = 0; i < 4; i++) {
        snprintf(path, sizeof(path), "%s/%s", sourceDir, tokFiles[i]);
        if (readAll(path, &len) == NULL) continue;
        if (copyFileTo(path, vocabDir)) copied++;
    }

    fprintf(stderr, "prune: wrote %s (%.1f MB, %d rows of %d), copied %d vocab files\n",
            dstPath, dataBytes / (1024.0 * 1024.0), V, srcRows, copied);
    return 1;
}
