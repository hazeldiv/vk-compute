#include <stdlib.h>
#include <math.h>
#include "session.h"
#include "data.h"
#include "validation.h"

double compute() {
    session s = createSession();

    int M = 1;
    int N = 12288;
    int K = 4096;
    float* input = getData(4321, M, K);
    float* input2 = getData(5555, M, K);
    float* gamma = getData(58923, M, K);
    float* weight = getData(936, K, N);
    uint16_t* weightFP16 = getDataFP16(936, K, N);
    QuantizedData weightINT8 = getDataINT8(936, K, N);
    QuantizedData weightINT4 = getDataINT4(936, K, N);
    uint16_t* weight2FP16 = getDataFP16(1348, K, N);
    QuantizedData weight2INT8 = getDataINT8(1348, K, N);
    QuantizedData weight2INT4 = getDataINT4(1348, K, N);

    int softmax_n = 1000;
    float* softmax_x = getData(2335, 1, softmax_n);
    float* softmax_v = getData(6346, softmax_n, 256);

    int att_seq = 2048;
    int att_heads = 16;
    int att_kv_heads = 4;
    int att_dim = 256;
    float* att_q = getData(7777, 1, att_heads * att_dim);
    uint16_t* att_k = getDataFP16(8100, att_kv_heads * att_dim, att_seq);
    uint16_t* att_v = getDataFP16(9100, att_kv_heads * att_dim, att_seq);
    QuantizedData att_k_i8 = getDataINT8(8100, att_kv_heads * att_dim, att_seq);
    QuantizedData att_v_i8 = getDataINT8(9100, att_kv_heads * att_dim, att_seq);
    QuantizedData att_k_i4 = getDataINT4(8100, att_kv_heads * att_dim, att_seq);
    QuantizedData att_v_i4 = getDataINT4(9100, att_kv_heads * att_dim, att_seq);

    int qkv_heads = 16;
    int qkv_kv_heads = 4;
    int qkv_dim = 256;
    int qkv_n = (qkv_heads + 2 * qkv_kv_heads) * qkv_dim;
    float* qkv_weight = getData(2468, K, qkv_n);
    uint16_t* qkv_weightFP16 = getDataFP16(2468, K, qkv_n);
    QuantizedData qkv_weightINT8 = getDataINT8(2468, K, qkv_n);
    QuantizedData qkv_weightINT4 = getDataINT4(2468, K, qkv_n);
    float* qkv_theta = (float*)malloc(sizeof(float) * (qkv_dim / 2));
    for (int i = 0; i < qkv_dim / 2; i++) {
        qkv_theta[i] = pow(1e6, -((double)i) / (qkv_dim / 2));
    }

    int w_in_n = 12320;
    int wo_n = 4096;
    uint16_t* w_inFP16 = getDataFP16(3579, K, w_in_n);
    QuantizedData w_inINT8 = getDataINT8(3579, K, w_in_n);
    QuantizedData w_inINT4 = getDataINT4(3579, K, w_in_n);
    uint16_t* woFP16 = getDataFP16(8642, K, wo_n);
    QuantizedData woINT8 = getDataINT8(8642, K, wo_n);
    QuantizedData woINT4 = getDataINT4(8642, K, wo_n);

    validateGEMV(s, M, N, K, input, weight);
    validateGEMVFP16(s, M, N, K, input, weightFP16);
    validateGEMVINT8(s, M, N, K, input, weightINT8);
    validateGEMVINT4(s, M, N, K, input, weightINT4);
    validateRmsNormGEMVFP16(s, M, N, K, input, gamma, weightFP16);
    validateRmsNormGEMVINT8(s, M, N, K, input, gamma, weightINT8);
    validateRmsNormGEMVINT4(s, M, N, K, input, gamma, weightINT4);
    validateGemvAddFP16(s, M, wo_n, K, input2, input, woFP16);
    validateGemvAddINT8(s, M, wo_n, K, input2, input, woINT8);
    validateGemvAddINT4(s, M, wo_n, K, input2, input, woINT4);
    validateRmsNormSwigluFfn(s, M, N, K, input, gamma, weight);
    validateRmsNormSwigluFfnFP16(s, M, N, K, input, gamma, weightFP16, weight2FP16);
    validateRmsNormSwigluFfnINT8(s, M, N, K, input, gamma, weightINT8, weight2INT8);
    validateRmsNormSwigluFfnINT4(s, M, N, K, input, gamma, weightINT4, weight2INT4);
    validateOnlineSoftmax(s, softmax_n, softmax_x, softmax_v);
    validateAttentionFP16(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionINT8(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k_i8, att_v_i8);
    validateAttentionINT4(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k_i4, att_v_i4);
    validateQkvRopeFP16(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightFP16, qkv_theta);
    validateQkvRopeINT8(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightINT8, qkv_theta);
    validateQkvRopeINT4(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightINT4, qkv_theta);
    validateGatedDeltaNetFP16(s, K, input, input2, gamma, w_inFP16, woFP16);
    validateGatedDeltaNetINT8(s, K, input, input2, gamma, w_inINT8, woINT8);
    validateGatedDeltaNetINT4(s, K, input, input2, gamma, w_inINT4, woINT4);

    free(input);
    free(input2);
    free(gamma);
    free(weight);
    free(weightFP16);
    free_quantized_data(weightINT8);
    free_quantized_data(weightINT4);
    free(weight2FP16);
    free_quantized_data(weight2INT8);
    free_quantized_data(weight2INT4);
    free(softmax_x);
    free(softmax_v);
    free(att_q);
    free(att_k);
    free(att_v);
    free_quantized_data(att_k_i8);
    free_quantized_data(att_v_i8);
    free_quantized_data(att_k_i4);
    free_quantized_data(att_v_i4);
    free(qkv_weight);
    free(qkv_weightFP16);
    free_quantized_data(qkv_weightINT8);
    free_quantized_data(qkv_weightINT4);
    free(qkv_theta);
    free(w_inFP16);
    free_quantized_data(w_inINT8);
    free_quantized_data(w_inINT4);
    free(woFP16);
    free_quantized_data(woINT8);
    free_quantized_data(woINT4);

    destroySession(s);
    return 0;
}
