#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "session.h"
#include "buffer.h"
#include "dispatch.h"
#include "data.h"

void transpose_block4(const float *input, float *output, int M, int N) {
    int out_N = N * 4;
    for (int b = 0; b < M; b += 4) {
        int out_row = b / 4;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < 4; ++k) {
                int in_idx = (b + k) * N + j;
                int out_idx = out_row * out_N + (j * 4) + k;
                output[out_idx] = input[in_idx];
            }
        }
    }
}


double compute() {
    int M = 1;
    int N = 4096;
    int K = 4096;
    float* input = getData(4321, M, K);
    float* gamma = getData(58923, M, K);
    float* weight = getData(936, K, N);
    float* transposedBlock4Weight = (float*)malloc(sizeof(float) * K * N);
    transpose_block4(weight, transposedBlock4Weight, K, N);
    float* output = (float*)malloc(sizeof(float) * M * N);
    memset(output, 0, sizeof(float) * M * N);

    session session = createSession();

    buffer inputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_VRAM);
    buffer gammaBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_VRAM);
    buffer weightBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedBlock4Weight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer outputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, output, sizeof(float) * M * N, MEMORY_VRAM);

    buffer buffers[] = {inputBuffer, gammaBuffer, weightBuffer, outputBuffer};
    createTransferAndCopy(session.dev.device, session.dev.queue, buffers, 4);

    operation ops[] = {
        // {
        //     .shader = "RmsNorm-GEMV.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBuffer, outputBuffer},
        //     .bufferCount = 4,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (K + 255) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        {
            .shader = "gemv4.spv",
            .buffers = {inputBuffer, weightBuffer, outputBuffer},
            .bufferCount = 3,
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

    int idx = 0;
    // float rms = 0.0f;
    // for (int i = 0; i < K; i++) rms += input[i] * input[i];
    // rms = sqrt(rms / (float)K) + 1e-5;

    // float result1[K];
    // for (int j = 0; j < K; j++) {
    //     result1[j] = 0.0f;
    //     for (int i = 0; i < K; i++) {
    //         result1[j] += (input[i] / rms * gamma[i]) * weight[i*K + j];
    //     }
    // }

    float result2 = 0.0f;
    for (int i = 0; i < K; i++) {
        result2 += input[i] * weight[i*N + idx];
    }

    printf("Output from index %d: %f %f\n", idx, outputVal[idx], result2);

    destroyBuffer(session.dev.device, inputBuffer);
    destroyBuffer(session.dev.device, gammaBuffer);
    destroyBuffer(session.dev.device, weightBuffer);
    destroyBuffer(session.dev.device, outputBuffer);
    destroySession(session);

    free(outputVal);
    free(input);
    free(gamma);
    free(weight);
    free(output);
    free(transposedBlock4Weight);

    return elapsedMs;
}