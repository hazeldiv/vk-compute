#include <vulkan/vulkan.h>
#include "dispatch.h"
#include "descriptor.h"
#include "pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PIPE_CACHE_MAX 64

typedef struct pipe_entry {
    char shader[128];
    int pushSize;
    int bufCount;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout setLayout;
} pipe_entry;

static pipe_entry pipeCache[PIPE_CACHE_MAX];
static int pipeCacheCount = 0;

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

void execute(session s, operation ops[], int opCount) {
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
        vkCmdDispatch(s.buffer, (uint32_t)op->dispatchX, (uint32_t)op->dispatchY, (uint32_t)op->dispatchZ);
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
}

double getExecutionTime(session s) {
    uint64_t timestamps[2];
    vkGetQueryPoolResults(s.dev.device, s.qpool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(s.dev.physicalDevice, &properties);
    double timestampPeriod = properties.limits.timestampPeriod;
    return (double)((unsigned int)timestamps[1] - (unsigned int)timestamps[0]) * timestampPeriod / 1000000.0;
}