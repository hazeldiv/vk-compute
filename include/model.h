#ifndef model_h
#define model_h

#include "data.h"

#define MODEL_LAYERS 32
#define MODEL_K 4096
#define MODEL_QKV_N 6144
#define MODEL_PROJ_N 12320
#define MODEL_FFN_N 12288
#define MODEL_HEADS 16
#define MODEL_KV_HEADS 4
#define MODEL_HEAD_DIM 256
#define MODEL_N_QK 16
#define MODEL_N_V 32
#define MODEL_DIM 128
#define MODEL_VOCAB 81920
#define MODEL_MAX_CTX 32768
#define MODEL_KV_ROWS (MODEL_KV_HEADS * MODEL_HEAD_DIM)
#define MODEL_Q_OFF (MODEL_HEADS * MODEL_HEAD_DIM)
#define MODEL_V_OFF ((MODEL_HEADS + MODEL_KV_HEADS) * MODEL_HEAD_DIM)
#define MODEL_MAX_GEMM 64
#define MODEL_MAX_OPS 320

typedef enum {
    ATTENTION_NONE,
    ATTENTION_FULL,
    ATTENTION_DELTA
} attention_type;

typedef struct attention {
    attention_type type;
    QuantType q;
} attention;

typedef enum {
    FFN_NONE,
    FFN_SWIGLU
} ffn_type;

typedef struct ffn {
    ffn_type type;
    QuantType q;
} ffn;

typedef struct layer {
    attention attn;
    ffn ffn;
} layer;

typedef struct model_config {
    const char* name;
    int layerCount;
    layer layers[MODEL_LAYERS];
    QuantType embedQ;
    QuantType lmHeadQ;
} model_config;

const char* model_shader(const char* base, QuantType q);

#endif