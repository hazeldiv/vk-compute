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

    dispatchContainer container = createDispatchContainer(dev, bufferCount, buffers, 3, M == 1 ? "gemv2.spv" : "gemm.spv");

    int pushConstants[] = {M, N, K};
    startDispatch(container.command);
    if (M == 1) {
        printf("Dispatching GEMV with dimensions M=%d, N=%d, K=%d\n", M, N, K);
        dispatch(container.descriptor, container.pipeline, container.command, (K + ts - 1)/ts,1,1, 3, pushConstants);
    } else {
        printf("Dispatching GEMM with dimensions M=%d, N=%d, K=%d\n", M, N, K);
        dispatch(container.descriptor, container.pipeline, container.command, (N + ts - 1)/ts,(M + ts - 1)/ts,1, 3, pushConstants);
    }
    endDispatch(container.command);

    VkFence fence = createFence(dev, container.command);

    uint64_t timestamps[2];
    vkGetQueryPoolResults(dev.device, container.command.queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(dev.physicalDevice, &properties);
    float timestampPeriod = properties.limits.timestampPeriod;
    double elapsedMs = (double)((unsigned long)timestamps[1] - (unsigned long)timestamps[0]) * timestampPeriod / 1000000.0;
    printf("Shader execution time: %.3f ms\n", elapsedMs);

    float* output = (float*)malloc(sizeof(float) * M * N);
    readBuffer(dev.device, dev.physicalDevice, dev.queue, bufferC, output);

    for (int i=0;i<bufferCount;i++) {
        destroyBuffer(dev.device, buffers[i]);
    }
    destroyContainer(dev, container.descriptor, container.pipeline, container.command, fence);
    free(output);
    free(A);
    free(B);
    free(C);
    
    return elapsedMs;
}
