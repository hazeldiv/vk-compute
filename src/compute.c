#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"
#include "descriptor.h"
#include "pipeline.h"
#include "command.h"
#include "dispatch.h"
#include "data.h"
#include "fence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double compute() {
    int M = 1;
    int N = 4096;
    int K = 4096;
    int ts = 256;
    float* input = getData(4321, M, K);
    float* gamma = getData(58923, M, K);
    float* weight = getData(936, K, N);
    float* output = (float*)malloc(sizeof(float) * M * K);
    memset(output, 0, sizeof(float) * M * N);

    device dev = createDevice();
    buffer inputBuffer = createBuffer(dev.device, dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_VRAM);
    buffer gammaBuffer = createBuffer(dev.device, dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_VRAM);
    buffer weightBuffer = createBuffer(dev.device, dev.physicalDevice, weight, sizeof(float) * K * N, MEMORY_VRAM);
    buffer outputBuffer = createBuffer(dev.device, dev.physicalDevice, output, sizeof(float) * M * K, MEMORY_VRAM);
    int bufferCount = 4;
    buffer buffers[] = {inputBuffer, gammaBuffer, weightBuffer, outputBuffer};

    createTransferAndCopy(dev.device, dev.queue, buffers, bufferCount);

    dispatchContainer container = createDispatchContainer(dev, bufferCount, buffers, 3, "RmsNorm-GEMV.spv");
    
    int pushConstants[] = {M, N, K};
    startDispatch(container.command);
    dispatch(container.descriptor, container.pipeline, container.command, (K + ts - 1)/ts,1,1, 3, pushConstants);
    endDispatch(container.command);

    VkFence fence = createFence(dev, container.command);

    uint64_t timestamps[2];
    vkGetQueryPoolResults(dev.device, container.command.queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(dev.physicalDevice, &properties);
    float timestampPeriod = properties.limits.timestampPeriod;
    double elapsedMs = (double)((unsigned long)timestamps[1] - (unsigned long)timestamps[0]) * timestampPeriod / 1000000.0;
    printf("Shader execution time: %.3f ms\n", elapsedMs);

    float* outputVal = (float*)malloc(sizeof(float) * M * N);
    readBuffer(dev.device, dev.physicalDevice, dev.queue, outputBuffer, outputVal);


    int idx = 4090;
    float rms = 0.0f;
    for (int i=0;i<K;i++) {
        rms += input[i] * input[i];
    }
    rms = sqrt(rms/(float)K) + 1e-5;

    float acc = 0.0f;
    for (int i=0;i<K;i++) {
        acc += (input[i] / rms * gamma[i]) * weight[i*K + idx];
    }

    printf("Output from index %d: %f %f\n",idx, outputVal[idx], acc);

    for (int i=0;i<bufferCount;i++) {
        destroyBuffer(dev.device, buffers[i]);
    }
    destroyContainer(dev, container.descriptor, container.pipeline, container.command, fence);
    free(outputVal);
    free(input);
    free(gamma);
    free(weight);
    free(output);
    
    return elapsedMs;
}
