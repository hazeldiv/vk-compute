#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "session.h"
#include "buffer.h"
#include "dispatch.h"
#include "data.h"

float fp16_to_float(uint16_t h);
uint16_t float_to_fp16(float f);

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
    
    float* output = (float*)malloc(sizeof(float) * M * N);
    memset(output, 0, sizeof(float) * M * N);

    session session = createSession();

    buffer inputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_VRAM);
    buffer gammaBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_VRAM);
    buffer weightBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer weightBufferFP16 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeightFP16, sizeof(uint16_t) * K * N, MEMORY_VRAM);
    buffer weightBufferINT8 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeightINT8, sizeof(uint8_t) * K * N, MEMORY_VRAM);
    buffer scaleBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_VRAM);
    buffer zeroPointBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_VRAM);
    buffer gateBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer upBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer outputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, output, sizeof(float) * M * N, MEMORY_VRAM);

    buffer buffers[] = {inputBuffer, gammaBuffer, weightBufferFP16, weightBufferINT8, weightBuffer, gateBuffer, upBuffer, outputBuffer, scaleBuffer, zeroPointBuffer};
    createTransferAndCopy(session.dev.device, session.dev.queue, buffers, 10);

    operation ops[] = {
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
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferINT8, outputBuffer, scaleBuffer, zeroPointBuffer},
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
        //     .buffers = {inputBuffer, weightBufferINT8, outputBuffer, scaleBuffer, zeroPointBuffer},
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
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferFP16, weightBufferFP16, outputBuffer},
        //     .bufferCount = 5,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        {
            .shader = "RmsNorm-swiglu-ffn-INT8.spv",
            .buffers = {inputBuffer, gammaBuffer, weightBufferINT8, weightBufferINT8, outputBuffer, scaleBuffer, zeroPointBuffer},
            .bufferCount = 7,
            .pushConstants = {M, N, K},
            .pushConstantCount = 3,
            .dispatchX = (N + 256-1) / 256,
            .dispatchY = 1,
            .dispatchZ = 1
        },
    };
    execute(session, ops, 1);
    double elapsedMs = getExecutionTime(session);
    printf("Shader execution time: %.3f ms\n", elapsedMs);
    float* outputVal = (float*)malloc(sizeof(float) * M * N);
    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, outputBuffer, outputVal);

    int idx = 8000;
    float result = 0.0f;

    //gemv with rms norm
    // for (int i = 0; i < K; i++) {
    //     result += input[i] * weight[i*N + idx];
    // }

    //gemv with rms norm
    // float rms = 0.0f;
    // for (int i = 0; i < K; i++) rms += input[i] * input[i];
    // rms = sqrt(rms / (float)K) + 1e-5;
    // for (int i = 0; i < K; i++) {
    //     result += (input[i] * gamma[i]) / rms * weight[i*N + idx];
    // }

    //swiglu ffn up & gate with rms norm
    float gate = 0.0f;
    float up = 0.0f;
    float rms = 0.0f;
    for (int i = 0; i < K; i++) rms += input[i] * input[i];
    rms = sqrt(rms / (float)K) + 1e-5;
    for (int i = 0; i < K; i++) {
        gate += (input[i] * gamma[i]) / rms * weight[i*N + idx];
    }
    for (int i = 0; i < K; i++) {
        up += (input[i] * gamma[i]) / rms * weight[i*N + idx];
    }
    gate /= (1.0 + exp2(-gate * 1.44269504));
    result = gate * up;

    printf("Output from index %d: %f %f\n", idx, outputVal[idx], result);

    destroyBuffer(session.dev.device, inputBuffer);
    destroyBuffer(session.dev.device, gammaBuffer);
    destroyBuffer(session.dev.device, weightBuffer);
    destroyBuffer(session.dev.device, outputBuffer);
    destroyBuffer(session.dev.device, gateBuffer);
    destroyBuffer(session.dev.device, weightBufferFP16);
    destroyBuffer(session.dev.device, weightBufferINT8);
    destroySession(session);

    free(outputVal);
    free(input);
    free(gamma);
    free(weight);
    free(output);
    free(transposedWeight);
    free(transposedWeightFP16);
    free(transposedWeightINT8);
    free(weightFP16);

    return 0;
}