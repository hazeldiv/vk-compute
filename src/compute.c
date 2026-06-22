#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"
#include "descriptor.h"
#include "pipeline.h"
#include "command.h"
#include "container.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

VkFence createFence(device dev, command command);
float* getData(int seed, int M, int N);
void cleanup(device dev, buffer buf, descriptor desc, pipeline pipe, command cmd, VkFence fence);

container createVKContainer(device dev, int bufferCount, buffer buffer[bufferCount], int varCount, char shader[]) {
    container VkContainer = {0};
    VkContainer.device = dev;
    VkContainer.descriptor = createDescriptor(dev.device, bufferCount, buffer);
    VkContainer.pipeline = createPipeline(dev.device, VkContainer.descriptor.layout, shader, sizeof(int) * varCount);
    VkContainer.command = createCommand(VkContainer.device.device);

    return VkContainer;
}

double compute() {
    int M = 16;
    int N = 4096*3;
    int K = 4096*3;
    int ts = M == 1 ? 256 : 16;
    float* A = getData(4321, M, K);
    float* B = getData(58923, K, N);
    float* c = (float*)malloc(sizeof(float) * M * N);
    memset(c, 0, sizeof(float) * M * N);

    device dev = createDevice();
    buffer bufferA = createBuffer(dev.device, dev.physicalDevice, A, sizeof(float) * M * K);
    buffer bufferB = createBuffer(dev.device, dev.physicalDevice, B, sizeof(float) * K * N);
    buffer bufferC = createBuffer(dev.device, dev.physicalDevice, c, sizeof(float) * M * N);
    buffer buffers[] = {bufferA, bufferB, bufferC};
    container VkContainer = createVKContainer(dev, 3, buffers, 3, M == 1 ? "gemv.spv" : "gemm.spv");

    int pushConstants[] = {M, N, K};
    startDispatch(VkContainer.command);
    if (M == 1) {
        printf("Dispatching GEMV with dimensions M=%d, N=%d, K=%d\n", M, N, K);
        dispatch(VkContainer.descriptor, VkContainer.pipeline, VkContainer.command, (K + ts - 1)/ts,1,1, 3, pushConstants);
    } else {
        printf("Dispatching GEMM with dimensions M=%d, N=%d, K=%d\n", M, N, K);
        dispatch(VkContainer.descriptor, VkContainer.pipeline, VkContainer.command, (N + ts - 1)/ts,(M + ts - 1)/ts,1, 3, pushConstants);
    }
    endDispatch(VkContainer.command);

    VkFence fence = createFence(dev, VkContainer.command);

    uint64_t timestamps[2];
    vkGetQueryPoolResults(dev.device, VkContainer.command.queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(dev.physicalDevice, &properties);
    float timestampPeriod = properties.limits.timestampPeriod;
    double elapsedMs = (double)((unsigned long)timestamps[1] - (unsigned long)timestamps[0]) * timestampPeriod / 1000000.0;
    printf("Shader execution time: %.3f ms\n", elapsedMs);

    float* outputResults = (float*)bufferC.mappedMemory;

    cleanup(dev, bufferA, VkContainer.descriptor, VkContainer.pipeline, VkContainer.command, fence);
    return elapsedMs;
}

void cleanup(device dev, buffer buf, descriptor desc, pipeline pipe, command cmd, VkFence fence) {
    vkDestroyFence(dev.device, fence, NULL);
    destroyCommand(dev.device, cmd);
    destroyPipeline(dev.device, pipe);
    destroyDescriptor(dev.device, desc);
    destroyBuffer(dev.device, buf);
    destroyDevice(dev);
}
