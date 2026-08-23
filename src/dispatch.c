#include <vulkan/vulkan.h>
#include "dispatch.h"
#include "descriptor.h"
#include "pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PIPE_CACHE_MAX 96
#define TIMING_AGG_MAX 128

typedef struct pipe_entry {
    char shader[128];
    int pushSize;
    int bufCount;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout setLayout;
} pipe_entry;

typedef struct timing_agg {
    char shader[128];
    double totalMs;
    long calls;
} timing_agg;

static pipe_entry pipeCache[PIPE_CACHE_MAX];
static int pipeCacheCount = 0;

static FILE* timingFile = NULL;
static timing_agg aggTable[TIMING_AGG_MAX];
static int aggCount = 0;
static double grandTotalMs = 0.0;

static pipe_entry* getPipeEntry(session s, const operation* op) {
    int pushSize = sizeof(int) * op->pushConstantCount;
    for (int i = 0; i < pipeCacheCount; i++) {
        if (pipeCache[i].pushSize == pushSize &&
            pipeCache[i].bufCount == op->bufferCount &&
            strcmp(pipeCache[i].shader, op->shader) == 0) {
            return &pipeCache[i];
        }
    }
    if (pipeCacheCount >= PIPE_CACHE_MAX) return NULL;

    pipe_entry* e = &pipeCache[pipeCacheCount++];
    snprintf(e->shader, sizeof(e->shader), "%s", op->shader);
    e->pushSize = pushSize;
    e->bufCount = op->bufferCount;
    e->setLayout = createDescriptorSetLayout(s.dev.device, op->bufferCount);

    char shaderPath[160];
    snprintf(shaderPath, sizeof(shaderPath), "shader/%s", op->shader);
    pipeline p = createPipeline(s.dev.device, e->setLayout, shaderPath, (uint32_t)pushSize);
    e->pipeline = p.pipeline;
    e->layout = p.layout;
    return e;
}

static void ensureTimingLog(void) {
    if (timingFile != NULL) return;
    timingFile = fopen("timing_log.txt", "w");
    if (timingFile == NULL) {
        fprintf(stderr, "Error: Failed to open timing_log.txt\n");
        exit(EXIT_FAILURE);
    }
}

static timing_agg* findAgg(const char* shader) {
    for (int i = 0; i < aggCount; i++) {
        if (strcmp(aggTable[i].shader, shader) == 0) return &aggTable[i];
    }
    if (aggCount >= TIMING_AGG_MAX) return NULL;
    timing_agg* a = &aggTable[aggCount++];
    snprintf(a->shader, sizeof(a->shader), "%s", shader);
    a->totalMs = 0.0;
    a->calls = 0;
    return a;
}

static int compareAgg(const void* pa, const void* pb) {
    const timing_agg* a = (const timing_agg*)pa;
    const timing_agg* b = (const timing_agg*)pb;
    if (a->totalMs < b->totalMs) return 1;
    if (a->totalMs > b->totalMs) return -1;
    return 0;
}

static double getTimestampPeriod(VkPhysicalDevice physicalDevice) {
    static double cached = 0.0;
    static VkPhysicalDevice cachedDevice = VK_NULL_HANDLE;
    if (cachedDevice != physicalDevice) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        cached = (double)properties.limits.timestampPeriod;
        cachedDevice = physicalDevice;
    }
    return cached;
}

static void logOpTimes(session s, operation ops[], int opCount, const char* phase, int token) {
    uint64_t stamps[2 + 2 * opCount];
    int queryCount = 2 + 2 * opCount;
    vkGetQueryPoolResults(s.dev.device, s.qpool, 0, queryCount,
                          queryCount * sizeof(uint64_t), stamps,
                          sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    double period = getTimestampPeriod(s.dev.physicalDevice);
    ensureTimingLog();

    for (int i = 0; i < opCount; i++) {
        uint32_t ticks = (uint32_t)stamps[3 + 2 * i] - (uint32_t)stamps[2 + 2 * i];
        double ms = (double)ticks * period / 1000000.0;
        char layerStr[8];
        if (ops[i].layer >= 0) {
            snprintf(layerStr, sizeof(layerStr), "%3d", ops[i].layer);
        } else {
            snprintf(layerStr, sizeof(layerStr), " --");
        }
        fprintf(timingFile, "[%-7s][tok=%5d][L=%3s] %-38s %10.3f ms\n",
                phase, token, layerStr, ops[i].shader, ms);

        timing_agg* a = findAgg(ops[i].shader);
        if (a != NULL) {
            a->totalMs += ms;
            a->calls++;
        }
        grandTotalMs += ms;
    }
    fflush(timingFile);
}

void closeTimingLog(void) {
    if (timingFile == NULL) return;
    qsort(aggTable, aggCount, sizeof(timing_agg), compareAgg);
    fprintf(timingFile, "\n==== timing summary ====\n");
    for (int i = 0; i < aggCount; i++) {
        double avg = aggTable[i].calls > 0 ? aggTable[i].totalMs / aggTable[i].calls : 0.0;
        fprintf(timingFile, "%-42s calls=%5ld total=%12.3f ms avg=%9.3f ms\n",
                aggTable[i].shader, aggTable[i].calls, aggTable[i].totalMs, avg);
    }
    fprintf(timingFile, "grand total: %.3f ms\n", grandTotalMs);
    fclose(timingFile);
    timingFile = NULL;
    aggCount = 0;
    grandTotalMs = 0.0;
}

static void runOps(session s, operation ops[], int opCount, const char* phase, int token) {
    vkWaitForFences(s.dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(s.dev.device, 1, &s.fence);
    vkResetCommandBuffer(s.buffer, 0);

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s.buffer, &beginInfo);

    vkCmdResetQueryPool(s.buffer, s.qpool, 0, TIMESTAMP_QUERY_COUNT);
    vkCmdWriteTimestamp(s.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.qpool, 0);

    descriptor descs[opCount];
    pipeline freshPipes[opCount];
    VkDescriptorSetLayout freshLayouts[opCount];
    int freshCount = 0;

    for (int i = 0; i < opCount; i++) {
        operation* op = &ops[i];

        pipe_entry* entry = getPipeEntry(s, op);
        VkPipelineLayout layout;
        VkPipeline pipe;
        VkDescriptorSetLayout setLayout;
        if (entry != NULL) {
            layout = entry->layout;
            pipe = entry->pipeline;
            setLayout = entry->setLayout;
        } else {
            setLayout = createDescriptorSetLayout(s.dev.device, op->bufferCount);
            char shaderPath[160];
            snprintf(shaderPath, sizeof(shaderPath), "shader/%s", op->shader);
            freshPipes[freshCount] = createPipeline(s.dev.device, setLayout, shaderPath, sizeof(int) * op->pushConstantCount);
            freshLayouts[freshCount] = setLayout;
            layout = freshPipes[freshCount].layout;
            pipe = freshPipes[freshCount].pipeline;
            freshCount++;
        }

        descs[i] = createDescriptorSetFromLayout(s.dev.device, setLayout, op->bufferCount, op->buffers);

        vkCmdBindPipeline(s.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(s.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descs[i].set, 0, NULL);

        if (op->pushConstantCount > 0) {
            vkCmdPushConstants(s.buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(int) * op->pushConstantCount, op->pushConstants);
        }
        if (i > 0) {
            VkMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(s.buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, NULL, 0, NULL);
        }
        vkCmdWriteTimestamp(s.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.qpool, 2 + 2 * i);
        vkCmdDispatch(s.buffer, (uint32_t)op->dispatchX, (uint32_t)op->dispatchY, (uint32_t)op->dispatchZ);
        vkCmdWriteTimestamp(s.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.qpool, 3 + 2 * i);
    }

    vkCmdWriteTimestamp(s.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.qpool, 1);
    vkEndCommandBuffer(s.buffer);

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &s.buffer;

    vkQueueSubmit(s.dev.queue, 1, &submitInfo, s.fence);
    vkWaitForFences(s.dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);

    for (int i = 0; i < opCount; i++) {
        destroyDescriptorSet(s.dev.device, descs[i]);
    }
    for (int i = 0; i < freshCount; i++) {
        destroyPipeline(s.dev.device, freshPipes[i]);
        vkDestroyDescriptorSetLayout(s.dev.device, freshLayouts[i], NULL);
    }

    if (phase != NULL) {
        logOpTimes(s, ops, opCount, phase, token);
    }
}

void execute(session s, operation ops[], int opCount) {
    runOps(s, ops, opCount, NULL, 0);
}

void executeLogged(session s, operation ops[], int opCount, const char* phase, int token) {
    runOps(s, ops, opCount, phase, token);
}

double getExecutionTime(session s) {
    uint64_t timestamps[2];
    vkGetQueryPoolResults(s.dev.device, s.qpool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    double timestampPeriod = getTimestampPeriod(s.dev.physicalDevice);
    return (double)((unsigned int)timestamps[1] - (unsigned int)timestamps[0]) * timestampPeriod / 1000000.0;
}
