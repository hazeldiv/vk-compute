#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "session.h"
#include "buffer.h"
#include "dispatch.h"
#include "data.h"

float fp16_to_float(uint16_t h);
uint16_t float_to_fp16(float f);

// Mirrors the shader's tiled online-softmax algorithm (tiles of 256, running
// max/sum rescale) instead of a single global pass, for validating that path.
void validate_softmax_v(const float *x, const float *v, float *o, int n) {
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;
    float acc[256] = {0};

    for (int k = 0; k < n; k += 256) {
        int tile_size = (n - k < 256) ? (n - k) : 256;

        float tile_max = -FLT_MAX;
        for (int i = 0; i < tile_size; i++) {
            if (x[k + i] > tile_max) tile_max = x[k + i];
        }

        float exp_vals[256];
        float tile_sum = 0.0f;
        for (int i = 0; i < tile_size; i++) {
            exp_vals[i] = expf(x[k + i] - tile_max);
            tile_sum += exp_vals[i];
        }

        if (tile_sum > 0.0f) {
            float new_max = fmaxf(max_val, tile_max);
            float scale_prev = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
            float scale_tile = expf(tile_max - new_max);

            sum_val = sum_val * scale_prev + tile_sum * scale_tile;

            for (int col = 0; col < 256; col++) {
                float tile_acc = 0.0f;
                for (int i = 0; i < tile_size; i++) {
                    tile_acc += exp_vals[i] * v[(k + i) * 256 + col];
                }
                acc[col] = acc[col] * scale_prev + tile_acc * scale_tile;
            }
            max_val = new_max;
        }
    }

    for (int col = 0; col < 256; col++) {
        o[col] = (sum_val > 0.0f) ? acc[col] / sum_val : 0.0f;
    }
}

void validate_attention(const float* q, const float* k, const float* v, float* o, int seq, int heads, int kv_heads, int dim) {
    for (int h = 0; h < heads; h++) {
        int kv = h / (heads / kv_heads);
        const float* qh = q + h * dim;
        const float* kh = k + kv * dim * seq;
        const float* vh = v + kv * dim * seq;

        float max_val = -FLT_MAX;
        float sum_val = 0.0f;
        float acc[256] = {0};

        for (int t = 0; t < seq; t += 256) {
            int tile = (seq - t < 256) ? (seq - t) : 256;

            float scores[256];
            float tile_max = -FLT_MAX;
            for (int i = 0; i < tile; i++) {
                float sc = 0.0f;
                for (int d = 0; d < dim; d++) sc += qh[d] * kh[d * seq + (t + i)];
                scores[i] = sc;
                if (sc > tile_max) tile_max = sc;
            }

            float expv[256];
            float tile_sum = 0.0f;
            for (int i = 0; i < tile; i++) {
                expv[i] = expf(scores[i] - tile_max);
                tile_sum += expv[i];
            }

            float new_max = fmaxf(max_val, tile_max);
            float scale_prev = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
            float scale_tile = expf(tile_max - new_max);
            sum_val = sum_val * scale_prev + tile_sum * scale_tile;

            for (int d = 0; d < dim; d++) {
                float tile_acc = 0.0f;
                for (int i = 0; i < tile; i++) tile_acc += expv[i] * vh[d * seq + (t + i)];
                acc[d] = acc[d] * scale_prev + tile_acc * scale_tile;
            }
            max_val = new_max;
        }

        for (int d = 0; d < dim; d++) o[h * dim + d] = (sum_val > 0.0f) ? acc[d] / sum_val : 0.0f;
    }
}

double compute() {
    int M = 1;
    int N = 1024 * 12;
    int K = 1024 * 4;
    float* input = getData(4321, M, K);
    float* gamma = getData(58923, M, K);
    float* weight = getData(936, K, N);
    float* transposedWeight = (float*)malloc(sizeof(float) * K * N);
    transpose_block16((uint8_t*)weight, (uint8_t*)transposedWeight, K, N, QUANT_FP32);

    uint16_t* weightFP16 = getDataFP16(936, K, N);
    uint16_t* transposedWeightFP16 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposedWeightFP16, K, N, QUANT_FP16);

    QuantizedData weightINT8 = getDataINT8(936, K, N);
    uint8_t* transposedWeightINT8 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposedWeightINT8, K, N, QUANT_INT8);

    QuantizedData weightINT4 = getDataINT4(936, K, N);
    uint8_t* transposedWeightINT4 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposedWeightINT4, K, N, QUANT_INT4);

    uint16_t* weight2FP16 = getDataFP16(1348, K, N);
    uint16_t* transposedWeight2FP16 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weight2FP16, (uint8_t*)transposedWeight2FP16, K, N, QUANT_FP16);

    QuantizedData weight2INT8 = getDataINT8(1348, K, N);
    uint8_t* transposedWeight2INT8 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weight2INT8.data, transposedWeight2INT8, K, N, QUANT_INT8);

    QuantizedData weight2INT4 = getDataINT4(1348, K, N);
    uint8_t* transposedWeight2INT4 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weight2INT4.data, transposedWeight2INT4, K, N, QUANT_INT4);

    int softmax_n = 1000;
    float* softmax_x = getData(2335, 1, softmax_n);
    float* softmax_v = getData(6346, softmax_n, 256);
    float* softmax_o = (float*)malloc(sizeof(float) * 1 * 256);
    memset(softmax_o, 0, sizeof(float) * 1 * 256);

    float* output = (float*)malloc(sizeof(float) * M * N);
    memset(output, 0, sizeof(float) * M * N);

    session session = createSession();
    buffer inputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_RAM);
    buffer weightBufferFP16 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeightFP16, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer weightBufferINT8 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeightINT8, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer weightBufferINT4 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeightINT4, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer scaleBufferINT8 = createBuffer(session.dev.device, session.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroPointBufferINT8 = createBuffer(session.dev.device, session.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer scaleBufferINT4 = createBuffer(session.dev.device, session.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroPointBufferINT4 = createBuffer(session.dev.device, session.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    
    buffer weightBuffer2FP16 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight2FP16, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer weightBuffer2INT8 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight2INT8, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer weightBuffer2INT4 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight2INT4, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer scaleBuffer2INT8 = createBuffer(session.dev.device, session.dev.physicalDevice, weight2INT8.scale, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer zeroPointBuffer2INT8 = createBuffer(session.dev.device, session.dev.physicalDevice, weight2INT8.z, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer scaleBuffer2INT4 = createBuffer(session.dev.device, session.dev.physicalDevice, weight2INT4.scale, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer zeroPointBuffer2INT4 = createBuffer(session.dev.device, session.dev.physicalDevice, weight2INT4.z, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);

    buffer gateBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_RAM);
    buffer outputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, output, sizeof(float) * M * N, MEMORY_RAM);

    buffer softmax_xBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, softmax_x, sizeof(float) * softmax_n, MEMORY_VRAM);
    buffer softmax_vBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, softmax_v, sizeof(float) * softmax_n * 256, MEMORY_VRAM);
    buffer softmax_oBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, softmax_o, sizeof(float) * 256, MEMORY_VRAM);

    int bufferCount = 16;
    buffer buffers[] = {
        inputBuffer, 
        gammaBuffer, 
        weightBufferFP16, 
        weightBufferINT8, 
        weightBufferINT4, 
        weightBuffer, 
        gateBuffer, 
        upBuffer, 
        outputBuffer, 
        scaleBufferINT8, 
        zeroPointBufferINT8, 
        scaleBufferINT4, 
        zeroPointBufferINT4,
        softmax_xBuffer,
        softmax_vBuffer,
        softmax_oBuffer};
    createTransferAndCopy(session.dev.device, session.dev.queue, buffers, bufferCount);

    int att_seq = 2048;
    int att_heads = 16;
    int att_kv_heads = 4;
    int att_dim = 256;

    float* att_q = getData(7777, 1, att_heads * att_dim);
    uint16_t* att_k_t = getDataFP16(8100, att_kv_heads * att_dim, att_seq);
    uint16_t* att_v_t = getDataFP16(9100, att_kv_heads * att_dim, att_seq);

    QuantizedData att_k_i8 = getDataINT8(8101, att_kv_heads * att_dim, att_seq);
    QuantizedData att_v_i8 = getDataINT8(9101, att_kv_heads * att_dim, att_seq);
    QuantizedData att_k_i4 = getDataINT4(8102, att_kv_heads * att_dim, att_seq);
    QuantizedData att_v_i4 = getDataINT4(9102, att_kv_heads * att_dim, att_seq);

    int att_rows = att_kv_heads * att_dim;
    int att_blocks = att_seq / 256;

    float* att_k_f32 = (float*)malloc(sizeof(float) * att_kv_heads * att_dim * att_seq);
    float* att_v_f32 = (float*)malloc(sizeof(float) * att_kv_heads * att_dim * att_seq);
    for (int i = 0; i < att_kv_heads * att_dim * att_seq; i++) att_k_f32[i] = fp16_to_float(att_k_t[i]);
    for (int i = 0; i < att_kv_heads * att_dim * att_seq; i++) att_v_f32[i] = fp16_to_float(att_v_t[i]);

    float* att_k_i8_f32 = (float*)malloc(sizeof(float) * att_kv_heads * att_dim * att_seq);
    float* att_v_i8_f32 = (float*)malloc(sizeof(float) * att_kv_heads * att_dim * att_seq);
    float* att_k_i4_f32 = (float*)malloc(sizeof(float) * att_kv_heads * att_dim * att_seq);
    float* att_v_i4_f32 = (float*)malloc(sizeof(float) * att_kv_heads * att_dim * att_seq);
    for (int i = 0; i < att_kv_heads * att_dim * att_seq; i++) {
        int row = i / att_seq;
        int tok = i % att_seq;
        int blk = tok / 256;
        int s_idx = blk * att_rows + row;
        att_k_i8_f32[i] = att_k_i8.data[i] * att_k_i8.scale[s_idx] - att_k_i8.z[s_idx];
        att_v_i8_f32[i] = att_v_i8.data[i] * att_v_i8.scale[s_idx] - att_v_i8.z[s_idx];
        uint8_t kb = att_k_i4.data[i / 2];
        uint8_t vb = att_v_i4.data[i / 2];
        int kn = (i & 1) ? (kb & 0x0F) : (kb >> 4);
        int vn = (i & 1) ? (vb & 0x0F) : (vb >> 4);
        att_k_i4_f32[i] = kn * att_k_i4.scale[s_idx] - att_k_i4.z[s_idx];
        att_v_i4_f32[i] = vn * att_v_i4.scale[s_idx] - att_v_i4.z[s_idx];
    }

    float* att_out = (float*)malloc(sizeof(float) * att_heads * att_dim);
    float* att_out_i8 = (float*)malloc(sizeof(float) * att_heads * att_dim);
    float* att_out_i4 = (float*)malloc(sizeof(float) * att_heads * att_dim);
    memset(att_out, 0, sizeof(float) * att_heads * att_dim);
    memset(att_out_i8, 0, sizeof(float) * att_heads * att_dim);
    memset(att_out_i4, 0, sizeof(float) * att_heads * att_dim);

    buffer attKeyBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, att_k_t, sizeof(uint16_t) * att_kv_heads * att_dim * att_seq, MEMORY_RAM);
    buffer attValueBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, att_v_t, sizeof(uint16_t) * att_kv_heads * att_dim * att_seq, MEMORY_RAM);
    buffer attQueryBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, att_q, sizeof(float) * att_heads * att_dim, MEMORY_RAM);
    buffer attOutBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, att_out, sizeof(float) * att_heads * att_dim, MEMORY_RAM);

    buffer attKeyBufferI8 = createBuffer(session.dev.device, session.dev.physicalDevice, att_k_i8.data, sizeof(uint8_t) * att_kv_heads * att_dim * att_seq, MEMORY_RAM);
    buffer attValueBufferI8 = createBuffer(session.dev.device, session.dev.physicalDevice, att_v_i8.data, sizeof(uint8_t) * att_kv_heads * att_dim * att_seq, MEMORY_RAM);
    buffer attKScaleI8 = createBuffer(session.dev.device, session.dev.physicalDevice, att_k_i8.scale, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attKZeroI8 = createBuffer(session.dev.device, session.dev.physicalDevice, att_k_i8.z, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attVScaleI8 = createBuffer(session.dev.device, session.dev.physicalDevice, att_v_i8.scale, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attVZeroI8 = createBuffer(session.dev.device, session.dev.physicalDevice, att_v_i8.z, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attOutBufferI8 = createBuffer(session.dev.device, session.dev.physicalDevice, att_out_i8, sizeof(float) * att_heads * att_dim, MEMORY_RAM);

    buffer attKeyBufferI4 = createBuffer(session.dev.device, session.dev.physicalDevice, att_k_i4.data, sizeof(uint8_t) * att_kv_heads * att_dim * att_seq / 2, MEMORY_RAM);
    buffer attValueBufferI4 = createBuffer(session.dev.device, session.dev.physicalDevice, att_v_i4.data, sizeof(uint8_t) * att_kv_heads * att_dim * att_seq / 2, MEMORY_RAM);
    buffer attKScaleI4 = createBuffer(session.dev.device, session.dev.physicalDevice, att_k_i4.scale, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attKZeroI4 = createBuffer(session.dev.device, session.dev.physicalDevice, att_k_i4.z, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attVScaleI4 = createBuffer(session.dev.device, session.dev.physicalDevice, att_v_i4.scale, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attVZeroI4 = createBuffer(session.dev.device, session.dev.physicalDevice, att_v_i4.z, sizeof(float) * att_rows * att_blocks, MEMORY_RAM);
    buffer attOutBufferI4 = createBuffer(session.dev.device, session.dev.physicalDevice, att_out_i4, sizeof(float) * att_heads * att_dim, MEMORY_RAM);

    operation ops[] = {
        // {
        //     .shader = "GEMV.spv",
        //     .buffers = {inputBuffer,  weightBuffer, outputBuffer},
        //     .bufferCount = 3,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 255) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "GEMV-INT4.spv",
        //     .buffers = {inputBuffer, weightBufferINT4, outputBuffer, scaleBufferINT4, zeroPointBufferINT4},
        //     .bufferCount = 5,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 255) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-GEMV-FP16.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferFP16, outputBuffer},
        //     .bufferCount = 4,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 255) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-GEMV-INT8.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferINT8, outputBuffer, scaleBufferINT8, zeroPointBufferINT8},
        //     .bufferCount = 6,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 255) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-GEMV-INT4.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferINT4, outputBuffer, scaleBufferINT4, zeroPointBufferINT4},
        //     .bufferCount = 6,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 255) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "GEMV-FP16.spv",
        //     .buffers = {inputBuffer, weightBufferFP16, outputBuffer},
        //     .bufferCount = 3,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 255) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "GEMV-INT8.spv",
        //     .buffers = {inputBuffer, weightBufferINT8, outputBuffer, scaleBufferINT8, zeroPointBufferINT8},
        //     .bufferCount = 5,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-swiglu-ffn.spv",
        //     .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outputBuffer},
        //     .bufferCount = 5,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-swiglu-ffn-FP16.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferFP16, weightBuffer2FP16, outputBuffer},
        //     .bufferCount = 5,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-swiglu-ffn-INT8.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferINT8, weightBuffer2INT8, outputBuffer, scaleBufferINT8, zeroPointBufferINT8, scaleBuffer2INT8, zeroPointBuffer2INT8},
        //     .bufferCount = 9,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-swiglu-ffn-INT4.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferINT4, weightBuffer2INT4, outputBuffer, scaleBufferINT4, zeroPointBufferINT4, scaleBuffer2INT4, zeroPointBuffer2INT4},
        //     .bufferCount = 9,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "online-softmax.spv",
        //     .buffers = {softmax_xBuffer, softmax_vBuffer, softmax_oBuffer},
        //     .bufferCount = 3,
        //     .pushConstants = {softmax_n},
        //     .pushConstantCount = 1,
        //     .dispatchX = 1,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        {
            .shader = "Att-full-FP16.spv",
            .buffers = {attKeyBuffer, attValueBuffer, attQueryBuffer, attOutBuffer},
            .bufferCount = 4,
            .pushConstants = {att_seq},
            .pushConstantCount = 1,
            .dispatchX = att_heads,
            .dispatchY = 1,
            .dispatchZ = 1
        },
        {
            .shader = "Att-full-INT8.spv",
            .buffers = {attKeyBufferI8, attValueBufferI8, attQueryBuffer, attOutBufferI8, attKScaleI8, attKZeroI8, attVScaleI8, attVZeroI8},
            .bufferCount = 8,
            .pushConstants = {att_seq},
            .pushConstantCount = 1,
            .dispatchX = att_heads,
            .dispatchY = 1,
            .dispatchZ = 1
        },
        {
            .shader = "Att-full-INT4.spv",
            .buffers = {attKeyBufferI4, attValueBufferI4, attQueryBuffer, attOutBufferI4, attKScaleI4, attKZeroI4, attVScaleI4, attVZeroI4},
            .bufferCount = 8,
            .pushConstants = {att_seq},
            .pushConstantCount = 1,
            .dispatchX = att_heads,
            .dispatchY = 1,
            .dispatchZ = 1
        },
    };
    execute(session, ops, 3);
    double elapsedMs = getExecutionTime(session);
    printf("Shader execution time: %.3f ms\n", elapsedMs);
    float* outputValSoftmax = (float*)malloc(sizeof(float) * 256);
    // float* outputVal = (float*)malloc(sizeof(float) * M * N);
    // float* outputVal_1 = (float*)malloc(sizeof(float) * K * N);
    // float* outputVal_2 = (float*)malloc(sizeof(float) * K * N);
    // readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, outputBuffer, outputVal);
    // readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, gateBuffer, outputVal_1);
    // readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, upBuffer, outputVal_2);
    // readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, softmax_oBuffer, outputValSoftmax);
    
    int idx = 0;
    float result = 0.0f;

    //gemv
    // for (int i = 0; i < K; i++) {
    //     result += input[i] * weight[i*N + idx];
    // }

    //gemv with pre rms norm
    // float rms = 0.0f;
    // for (int i = 0; i < K; i++) rms += input[i] * input[i];
    // rms = sqrt(rms / (float)K) + 1e-5;
    // for (int i = 0; i < K; i++) {
    //     result += (input[i] * gamma[i]) / rms * weight[i*N + idx];
    // }

    //swiglu ffn up & gate with pre rms norm
    // float gate = 0.0f;
    // float up = 0.0f;
    // float rms = 0.0f;
    // for (int i = 0; i < K; i++) rms += input[i] * input[i];
    // rms = sqrt(rms / (float)K) + 1e-5;
    // for (int i = 0; i < K; i++) {
    //     gate += (input[i] * gamma[i]) / rms * weight[i*N + idx];
    // }
    // for (int i = 0; i < K; i++) {
    //     up += (input[i] * gamma[i]) / rms * weight[i*N + idx];
    // }
    // gate /= (1.0 + exp2(-gate * 1.44269504));
    // result = gate * up;


    // gate = 0.0f;
    // up = 0.0f;
    // float quantizedResult = 0.0f;
    // for (int i = 0; i < K; i++) {
    //     float d_q = weightINT8.data[i*N+idx] * weightINT8.scale[i] - weightINT8.z[i];
    //     gate += (input[i] * gamma[i]) / rms * d_q;
    // }
    // for (int i = 0; i < K; i++) {
    //     float d_q = weightINT8.data[i*N+idx] * weightINT8.scale[i] - weightINT8.z[i];
    //     up += (input[i] * gamma[i]) / rms * d_q;
    // }
    // gate /= (1.0 + exp2(-gate * 1.44269504));
    // quantizedResult = gate * up;

    // gate = 0.0f;
    // up = 0.0f;
    // float quantizedResult = 0.0f;
    // for (int i = 0; i < K; i++) {
    //     int q = (weightINT4.data[(i*(N/2)+idx/2)] >> 4) & 0x0F;
    //     float d_q = q * weightINT4.scale[i] - weightINT4.z[i];
    //     gate += (input[i] * gamma[i]) / rms * d_q;
    // }
    // for (int i = 0; i < K; i++) {
    //     int q = (weightINT4.data[(i*(N/2)+idx/2)] >> 4) & 0x0F;
    //     float d_q = q * weightINT4.scale[i] - weightINT4.z[i];
    //     up += (input[i] * gamma[i]) / rms * d_q;
    // }
    // gate /= (1.0 + exp2(-gate * 1.44269504));
    // quantizedResult = gate * up;


    // printf("%d %f\n", weightINT4.data[0] >> 4, weightINT4.scale[0]);

    // float quantizedResult = 0.0f;
    // for (int i = 0; i < K; i++) {
    //     int q = (weightINT4.data[(i*(N/2)+idx/2)] >> 4) & 0x0F;
    //     float d_q = q * weightINT4.scale[i] - weightINT4.z[i];
    //     quantizedResult += input[i] * d_q;
    //     result += input[i] * weight[i*N + idx];
    // }
    // float quantizedResult = 0.0f;
    // for (int i = 0; i < K; i++) {
    //     int q = weightINT8.data[i*N+idx];
    //     float d_q = q * weightINT8.scale[i] - weightINT8.z[i];
    //     quantizedResult += input[i] * d_q;
    //     result += input[i] * weight[i*N + idx];
    // }


    // printf("Output from index %d: %f %f\n", idx, outputVal[idx], result);
    // printf("Output from index %d: %f %f\n", idx, outputVal[idx], quantizedResult);

    // destroyBuffer(session.dev.device, inputBuffer);
    // destroyBuffer(session.dev.device, gammaBuffer);
    // destroyBuffer(session.dev.device, weightBuffer);
    // destroyBuffer(session.dev.device, outputBuffer);
    // destroyBuffer(session.dev.device, gateBuffer);
    // destroyBuffer(session.dev.device, weightBufferFP16);
    // destroyBuffer(session.dev.device, weightBufferINT8);
    // destroyBuffer(session.dev.device, weightBufferINT4);
    // destroyBuffer(session.dev.device, scaleBufferINT8);
    // destroyBuffer(session.dev.device, zeroPointBufferINT8);
    // destroyBuffer(session.dev.device, scaleBufferINT4);
    // destroyBuffer(session.dev.device, zeroPointBufferINT4);
    // float output_val[256];
    // validate_softmax_v(softmax_x, softmax_v, output_val, softmax_n);
    // printf("Output from softmax: shader= %f validation= %f\n", outputValSoftmax[40], output_val[40]);

    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, attOutBuffer, att_out);
    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, attOutBufferI8, att_out_i8);
    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, attOutBufferI4, att_out_i4);

    float* att_ref = (float*)malloc(sizeof(float) * att_heads * att_dim);
    float* att_ref_i8 = (float*)malloc(sizeof(float) * att_heads * att_dim);
    float* att_ref_i4 = (float*)malloc(sizeof(float) * att_heads * att_dim);
    validate_attention(att_q, att_k_f32, att_v_f32, att_ref, att_seq, att_heads, att_kv_heads, att_dim);
    validate_attention(att_q, att_k_i8_f32, att_v_i8_f32, att_ref_i8, att_seq, att_heads, att_kv_heads, att_dim);
    validate_attention(att_q, att_k_i4_f32, att_v_i4_f32, att_ref_i4, att_seq, att_heads, att_kv_heads, att_dim);

    float att_max_err = 0.0f;
    for (int i = 0; i < att_heads * att_dim; i++) {
        float e = fabsf(att_out[i] - att_ref[i]);
        if (e > att_max_err) att_max_err = e;
    }
    float att_max_err_i8 = 0.0f;
    for (int i = 0; i < att_heads * att_dim; i++) {
        float e = fabsf(att_out_i8[i] - att_ref_i8[i]);
        if (e > att_max_err_i8) att_max_err_i8 = e;
    }
    float att_max_err_i4 = 0.0f;
    for (int i = 0; i < att_heads * att_dim; i++) {
        float e = fabsf(att_out_i4[i] - att_ref_i4[i]);
        if (e > att_max_err_i4) att_max_err_i4 = e;
    }

    printf("Attention FP16: shader[%d]= %f ref[%d]= %f max_err= %f\n", idx, att_out[idx], idx, att_ref[idx], att_max_err);
    printf("Attention INT8: shader[%d]= %f ref[%d]= %f max_err= %f\n", idx, att_out_i8[idx], idx, att_ref_i8[idx], att_max_err_i8);
    printf("Attention INT4: shader[%d]= %f ref[%d]= %f max_err= %f\n", idx, att_out_i4[idx], idx, att_ref_i4[idx], att_max_err_i4);

    for (int i = 0; i < bufferCount; i++) {
        destroyBuffer(session.dev.device, buffers[i]);
    }

    destroyBuffer(session.dev.device, attKeyBuffer);
    destroyBuffer(session.dev.device, attValueBuffer);
    destroyBuffer(session.dev.device, attQueryBuffer);
    destroyBuffer(session.dev.device, attOutBuffer);
    destroyBuffer(session.dev.device, attKeyBufferI8);
    destroyBuffer(session.dev.device, attValueBufferI8);
    destroyBuffer(session.dev.device, attKScaleI8);
    destroyBuffer(session.dev.device, attKZeroI8);
    destroyBuffer(session.dev.device, attVScaleI8);
    destroyBuffer(session.dev.device, attVZeroI8);
    destroyBuffer(session.dev.device, attOutBufferI8);
    destroyBuffer(session.dev.device, attKeyBufferI4);
    destroyBuffer(session.dev.device, attValueBufferI4);
    destroyBuffer(session.dev.device, attKScaleI4);
    destroyBuffer(session.dev.device, attKZeroI4);
    destroyBuffer(session.dev.device, attVScaleI4);
    destroyBuffer(session.dev.device, attVZeroI4);
    destroyBuffer(session.dev.device, attOutBufferI4);

    // free(outputVal);
    free(input);
    free(gamma);
    free(weight);
    free(output);
    free(transposedWeight);
    free(transposedWeightFP16);
    free(transposedWeightINT8);
    free(transposedWeightINT4);
    free(weightFP16);
    free_quantized_data(weightINT8);
    free_quantized_data(weightINT4);
    free(softmax_x);
    free(softmax_v);
    free(softmax_o);
    free(outputValSoftmax);
    free(att_q);
    free(att_k_t);
    free(att_v_t);
    free_quantized_data(att_k_i8);
    free_quantized_data(att_v_i8);
    free_quantized_data(att_k_i4);
    free_quantized_data(att_v_i4);
    free(att_k_f32);
    free(att_v_f32);
    free(att_k_i8_f32);
    free(att_v_i8_f32);
    free(att_k_i4_f32);
    free(att_v_i4_f32);
    free(att_out);
    free(att_out_i8);
    free(att_out_i4);
    free(att_ref);
    free(att_ref_i8);
    free(att_ref_i4);
    return 0;
}
