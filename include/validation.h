#ifndef validation_h
#define validation_h

#include "session.h"
#include "data.h"

typedef struct {
    int M;
    int N;
    int K;
    float* input;
    float* gamma;
    float* weight;
    uint16_t* weightFP16;
    uint16_t* weight2FP16;
    QuantizedData weightINT8;
    QuantizedData weightINT4;
    QuantizedData weight2INT8;
    QuantizedData weight2INT4;
    int softmax_n;
    float* softmax_x;
    float* softmax_v;
    int att_seq;
    int att_heads;
    int att_kv_heads;
    int att_dim;
    float* att_q;
    uint16_t* att_k;
    uint16_t* att_v;
    QuantizedData att_k_i8;
    QuantizedData att_v_i8;
    QuantizedData att_k_i4;
    QuantizedData att_v_i4;
} ValidationData;

void validateGEMV(session s, const ValidationData* d);
void validateGEMVFP16(session s, const ValidationData* d);
void validateGEMVINT8(session s, const ValidationData* d);
void validateGEMVINT4(session s, const ValidationData* d);
void validateRmsNormGEMVFP16(session s, const ValidationData* d);
void validateRmsNormGEMVINT8(session s, const ValidationData* d);
void validateRmsNormGEMVINT4(session s, const ValidationData* d);
void validateRmsNormSwigluFfn(session s, const ValidationData* d);
void validateRmsNormSwigluFfnFP16(session s, const ValidationData* d);
void validateRmsNormSwigluFfnINT8(session s, const ValidationData* d);
void validateRmsNormSwigluFfnINT4(session s, const ValidationData* d);
void validateOnlineSoftmax(session s, const ValidationData* d);
void validateAttentionFP16(session s, const ValidationData* d);
void validateAttentionINT8(session s, const ValidationData* d);
void validateAttentionINT4(session s, const ValidationData* d);

#endif
