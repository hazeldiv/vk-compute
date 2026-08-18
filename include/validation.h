#ifndef validation_h
#define validation_h

#include "session.h"
#include "data.h"

void validateGEMV(session s, int M, int N, int K, float* input, float* weight);
void validateGEMVFP16(session s, int M, int N, int K, float* input, uint16_t* weightFP16);
void validateGEMVINT8(session s, int M, int N, int K, float* input, QuantizedData weightINT8);
void validateGEMVINT4(session s, int M, int N, int K, float* input, QuantizedData weightINT4);
void validateRmsNormGEMVFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16);
void validateRmsNormGEMVINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8);
void validateRmsNormGEMVINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4);
void validateRmsNormSwigluFfn(session s, int M, int N, int K, float* input, float* gamma, float* weight);
void validateRmsNormSwigluFfnFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16);
void validateRmsNormSwigluFfnINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8);
void validateRmsNormSwigluFfnINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4);
void validateOnlineSoftmax(session s, int softmax_n, float* softmax_x, float* softmax_v);
void validateAttentionFP16(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v);
void validateAttentionINT8(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, QuantizedData att_k_i8, QuantizedData att_v_i8);
void validateAttentionINT4(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, QuantizedData att_k_i4, QuantizedData att_v_i4);
void validateQkvRopeFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta);
void validateQkvRopeINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta);
void validateQkvRopeINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta);
void validateGatedDeltaNetFP16(session s, int K, float* input, float* input2, float* gamma, uint16_t* w_inFP16, uint16_t* woFP16);
void validateGatedDeltaNetINT8(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT8, QuantizedData woINT8);
void validateGatedDeltaNetINT4(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT4, QuantizedData woINT4);

#endif
