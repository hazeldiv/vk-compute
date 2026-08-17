#include <stdlib.h>
#include "session.h"
#include "data.h"
#include "validation.h"

double compute() {
    session s = createSession();

    ValidationData d = {0};
    d.M = 1;
    d.N = 12288;
    d.K = 4096;
    d.input = getData(4321, d.M, d.K);
    d.gamma = getData(58923, d.M, d.K);
    d.weight = getData(936, d.K, d.N);
    d.weightFP16 = getDataFP16(936, d.K, d.N);
    d.weightINT8 = getDataINT8(936, d.K, d.N);
    d.weightINT4 = getDataINT4(936, d.K, d.N);
    d.weight2FP16 = getDataFP16(1348, d.K, d.N);
    d.weight2INT8 = getDataINT8(1348, d.K, d.N);
    d.weight2INT4 = getDataINT4(1348, d.K, d.N);

    d.softmax_n = 1000;
    d.softmax_x = getData(2335, 1, d.softmax_n);
    d.softmax_v = getData(6346, d.softmax_n, 256);

    d.att_seq = 2048;
    d.att_heads = 16;
    d.att_kv_heads = 4;
    d.att_dim = 256;
    d.att_q = getData(7777, 1, d.att_heads * d.att_dim);
    d.att_k = getDataFP16(8100, d.att_kv_heads * d.att_dim, d.att_seq);
    d.att_v = getDataFP16(9100, d.att_kv_heads * d.att_dim, d.att_seq);
    d.att_k_i8 = getDataINT8(8100, d.att_kv_heads * d.att_dim, d.att_seq);
    d.att_v_i8 = getDataINT8(9100, d.att_kv_heads * d.att_dim, d.att_seq);
    d.att_k_i4 = getDataINT4(8100, d.att_kv_heads * d.att_dim, d.att_seq);
    d.att_v_i4 = getDataINT4(9100, d.att_kv_heads * d.att_dim, d.att_seq);

    validateGEMV(s, &d);
    validateGEMVFP16(s, &d);
    validateGEMVINT8(s, &d);
    validateGEMVINT4(s, &d);
    validateRmsNormGEMVFP16(s, &d);
    validateRmsNormGEMVINT8(s, &d);
    validateRmsNormGEMVINT4(s, &d);
    validateRmsNormSwigluFfn(s, &d);
    validateRmsNormSwigluFfnFP16(s, &d);
    validateRmsNormSwigluFfnINT8(s, &d);
    validateRmsNormSwigluFfnINT4(s, &d);
    validateOnlineSoftmax(s, &d);
    validateAttentionFP16(s, &d);
    validateAttentionINT8(s, &d);
    validateAttentionINT4(s, &d);

    free(d.input);
    free(d.gamma);
    free(d.weight);
    free(d.weightFP16);
    free_quantized_data(d.weightINT8);
    free_quantized_data(d.weightINT4);
    free(d.weight2FP16);
    free_quantized_data(d.weight2INT8);
    free_quantized_data(d.weight2INT4);
    free(d.softmax_x);
    free(d.softmax_v);
    free(d.att_q);
    free(d.att_k);
    free(d.att_v);
    free_quantized_data(d.att_k_i8);
    free_quantized_data(d.att_v_i8);
    free_quantized_data(d.att_k_i4);
    free_quantized_data(d.att_v_i4);

    destroySession(s);
    return 0;
}
