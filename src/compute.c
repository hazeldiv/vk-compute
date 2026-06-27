#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"
#include "descriptor.h"
#include "pipeline.h"
#include "command.h"
#include "dispatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

VkFence createFence(device dev, command command);
float* getData(int seed, int M, int N);
void cleanup(device dev, int bufferCount, buffer buffer[], descriptor desc, pipeline pipe, command cmd, VkFence fence);

void transpose(const float* src, float* dest, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int src_idx = i * n + j;
            int dest_idx = j * m + i;
            
            dest[dest_idx] = src[src_idx];
        }
    }
}

double compute() {
    int M = 1;
    int N = 4096 * 3;
    int K = 4096 * 3;
    int ts = M == 1 ? 256 : 16;
    float* A = getData(4321, M, K);
    float* B = getData(58923, K, N);
    float* C = (float*)malloc(sizeof(float) * M * N);
    memset(C, 0, sizeof(float) * M * N);

    device dev = createDevice();
    buffer bufferA = createBuffer(dev.device, dev.physicalDevice, A, sizeof(float) * M * K, MEMORY_VRAM);
    buffer bufferB = createBuffer(dev.device, dev.physicalDevice, B, sizeof(float) * K * N, MEMORY_VRAM);
    buffer bufferC = createBuffer(dev.device, dev.physicalDevice, C, sizeof(float) * M * N, MEMORY_VRAM);
    int bufferCount = 3;

    buffer buffers[] = {bufferA, bufferB, bufferC};
    createTransferAndCopy(dev.device, dev.queue, buffers, bufferCount);

    dispatchContainer VkContainer = createDispatchContainer(dev, bufferCount, buffers, 3, M == 1 ? "gemv2.spv" : "gemm.spv");

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

    float* output = (float*)malloc(sizeof(float) * M * N);
    readBuffer(dev.device, dev.physicalDevice, dev.queue, bufferC, output);

    cleanup(dev, bufferCount, buffers, VkContainer.descriptor, VkContainer.pipeline, VkContainer.command, fence);
    free(output);
    free(A);
    free(B);
    free(C);
    
    return elapsedMs;
}

void cleanup(device dev, int bufferCount, buffer buffer[], descriptor desc, pipeline pipe, command cmd, VkFence fence) {
    vkDestroyFence(dev.device, fence, NULL);
    destroyCommand(dev.device, cmd);
    destroyPipeline(dev.device, pipe);
    destroyDescriptor(dev.device, desc);
    for (int i=0;i<bufferCount;i++) {
        destroyBuffer(dev.device, buffer[i]);
    }
    destroyDevice(dev);
}
