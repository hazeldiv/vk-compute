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
    int N = 1024 * 4;
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
    printf("%d\n", weightINT4.data[0]);
    transpose_block16(weightINT4.data, transposedWeightINT4, K, N, QUANT_INT4);
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
    buffer gateBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, transposedWeight, sizeof(float) * K * N, MEMORY_RAM);
    buffer outputBuffer = createBuffer(session.dev.device, session.dev.physicalDevice, output, sizeof(float) * M * N, MEMORY_RAM);
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
        zeroPointBufferINT4};
    createTransferAndCopy(session.dev.device, session.dev.queue, buffers, 13);

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
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferFP16, weightBufferFP16, outputBuffer},
        //     .bufferCount = 5,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        // {
        //     .shader = "RmsNorm-swiglu-ffn-INT8.spv",
        //     .buffers = {inputBuffer, gammaBuffer, weightBufferINT8, weightBufferINT8, outputBuffer, scaleBufferINT8, zeroPointBufferINT8},
        //     .bufferCount = 7,
        //     .pushConstants = {M, N, K},
        //     .pushConstantCount = 3,
        //     .dispatchX = (N + 256-1) / 256,
        //     .dispatchY = 1,
        //     .dispatchZ = 1
        // },
        {
            .shader = "RmsNorm-swiglu-ffn-INT4.spv",
            .buffers = {inputBuffer, gammaBuffer, weightBufferINT4, weightBufferINT4, outputBuffer, scaleBufferINT4, zeroPointBufferINT4},
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
    float* outputVal_1 = (float*)malloc(sizeof(float) * K * N);
    float* outputVal_2 = (float*)malloc(sizeof(float) * K * N);
    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, outputBuffer, outputVal);
    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, gateBuffer, outputVal_1);
    readBuffer(session.dev.device, session.dev.physicalDevice, session.dev.queue, upBuffer, outputVal_2);
    
    int idx = 1000;
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

    // printf("%d %f\n", weightINT4.data[0] >> 4, weightINT4.scale[0]);

    // float quantizedResult = 0.0f;
    // for (int i = 0; i < K; i++) {
    //     int q = (weightINT4.data[(i*(N/2)+idx/2)] >> 4) & 0x0F;
    //     float d_q = q * weightINT4.scale[i] - weightINT4.z[i];
    //     quantizedResult += input[i] * d_q;
    //     result += input[i] * weight[i*N + idx];
    // }


    printf("Output from index %d: %f %f\n", idx, outputVal[idx], result);

    destroyBuffer(session.dev.device, inputBuffer);
    destroyBuffer(session.dev.device, gammaBuffer);
    destroyBuffer(session.dev.device, weightBuffer);
    destroyBuffer(session.dev.device, outputBuffer);
    destroyBuffer(session.dev.device, gateBuffer);
    destroyBuffer(session.dev.device, weightBufferFP16);
    destroyBuffer(session.dev.device, weightBufferINT8);
    destroyBuffer(session.dev.device, weightBufferINT4);
    destroyBuffer(session.dev.device, scaleBufferINT8);
    destroyBuffer(session.dev.device, zeroPointBufferINT8);
    destroyBuffer(session.dev.device, scaleBufferINT4);
    destroyBuffer(session.dev.device, zeroPointBufferINT4);

    free(outputVal);
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
    return 0;
}