#ifndef model_h
#define model_h

#include "data.h"

#define MODEL_LAYERS 6
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
#define MODEL_MAX_OPS 192

extern const int model_attn_layer[MODEL_LAYERS];
extern const QuantType model_layer_q[MODEL_LAYERS];
extern const QuantType model_ffn_q[MODEL_LAYERS];

int model_is_attention(int layer);
int model_attn_count(void);
const char* model_shader(const char* base, QuantType q);

#endif