#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "session.h"
#include "buffer.h"
#include "dispatch.h"
#include "data.h"


double compute() {
    int M = 1;
    int N = 1024 * 4;
    int K = 1024 * 4;
    float* input = getData(4321, M, K);
    float* gamma = getData(58923, M, K);
    float* weight = getData(936, K, N);
    float* transposedBlock4Weight = (float*)malloc(sizeof(float) * K * N);
    transpose_block4(weight, transposedBlock4Weight, K, N);

    uint16_t* weightFP16 = getDataFP16(936, K, N);
    uint16_t* transposedBlock4WeightFP16 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block4_FP16(weightFP16, transposedBlock4WeightFP16, K, N);

    float* output = (float*)malloc(sizeof(float) * M * N);
    memset(output, 0, sizeof(float) * M * N);

    session session = createSession();

    buffer inputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_VRAM);
    buffer gammaBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_VRAM);
    buffer weightBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedBlock4Weight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer weightBufferFP16 = createBuffer(session.dev.device, session.dev.physicalDevice, transposedBlock4WeightFP16, sizeof(uint16_t) * K * N, MEMORY_VRAM);
    buffer gateBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedBlock4Weight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer upBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedBlock4Weight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer outputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, output, sizeof(float) * M * N, MEMORY_VRAM);

    buffer buffers[] = {inputBuffer, gammaBuffer, weightBufferFP16, weightBuffer, gateBuffer, upBuffer, outputBuffer};
    createTransferAndCopy(session.dev.device, session.dev.queue, buffers, 6);
    // buffer buffers[] = {inputBuffer, gammaBuffer, weightBuffer, outputBuffer};
    // createTransferAndCopy(session.dev.device, session.dev.queue, buffers, 4);

    operation ops[] = {
        {
            .shader = "RmsNorm-GEMV-fp16.spv",
            .buffers = {inputBuffer, gammaBuffer, weightBufferFP16, outputBuffer},
            .bufferCount = 4,
            .pushConstants = {M, N, K},
            .pushConstantCount = 3,
            .dispatchX = (N + 255) / 256,
            .dispatchY = 1,
            .dispatchZ = 1
        },
        // {
        //     .shader = "gemv6.spv",
        //     .buffers = {inputBuffer, weightBuffer, outputBuffer},
        //     .bufferCount = 3,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "swiglu-ffn.spv",
        //     .buffers = {inputBuffer, gateBuffer, upBuffer, outputBuffer},
        //     .bufferCount = 4,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
    };
    execute(session, ops, 1);
    double elapsedMs = getExecutionTime(session);
    printf("Shader execution time: %.3f ms\n", elapsedMs);
    float* outputVal = (float*)malloc(sizeof(float) * M * N);
    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, outputBuffer, outputVal);

    int idx = 4000;
    float rms = 0.0f;
    for (int i = 0; i < K; i++) rms += input[i] * input[i];
    rms = sqrt(rms / (float)K) + 1e-5;

    // float result1[K];
    // for (int j = 0; j < K; j++) {
    //     result1[j] = 0.0f;
    //     for (int i = 0; i < K; i++) {
    //         result1[j] += (input[i] / rms * gamma[i]) * weight[i*K + j];
    //     }
    // }

    float result2 = 0.0f;
    // float down = 0.0f;
    for (int i = 0; i < K; i++) {
        result2 += (input[i] * gamma[i]) / rms * weight[i*N + idx] ;
    }

    // for (int i = 0; i < K; i++) {
    //     down += input[i] / rms * weight[i*N + idx];
    // }
    // result2 /= (1.0 + exp2(-result2 * 1.44269504));
    // result2 *= down;

    printf("Output from index %d: %f %f\n", idx, outputVal[idx], result2);

    destroyBuffer(session.dev.device, inputBuffer);
    destroyBuffer(session.dev.device, gammaBuffer);
    destroyBuffer(session.dev.device, weightBuffer);
    destroyBuffer(session.dev.device, outputBuffer);
    destroyBuffer(session.dev.device, gateBuffer);
    destroyBuffer(session.dev.device, upBuffer);
    destroySession(session);

    free(outputVal);
    free(input);
    free(gamma);
    free(weight);
    free(output);
    free(transposedBlock4Weight);
    free(transposedBlock4WeightFP16);
    free(weightFP16);

    return elapsedMs;
}