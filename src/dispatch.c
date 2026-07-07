#include <vulkan/vulkan.h>
#include "dispatch.h"
#include "descriptor.h"
#include "pipeline.h"
#include <stdio.h>
#include <stdlib.h>

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

    for (int i = 0; i < opCount; i++) {
        operation* op = &ops[0];

        descriptor desc = createDescriptor(s.dev.device, op->bufferCount, op->buffers);
        pipeline pipe = createPipeline(s.dev.device, desc.layout, op->shader, sizeof(int) * op->pushConstantCount);

        vkCmdBindPipeline(s.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(s.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &desc.set, 0, NULL);

        if (op->pushConstantCount > 0) {
            vkCmdPushConstants(s.buffer, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
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

        destroyPipeline(s.dev.device, pipe);
        destroyDescriptor(s.dev.device, desc);
    }

    vkCmdWriteTimestamp(s.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s.qpool, 1);
    vkEndCommandBuffer(s.buffer);

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &s.buffer;

    vkQueueSubmit(s.dev.queue, 1, &submitInfo, s.fence);
    vkWaitForFences(s.dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
}

double getExecutionTime(session s) {
    uint64_t timestamps[2];
    vkGetQueryPoolResults(s.dev.device, s.qpool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(s.dev.physicalDevice, &properties);
    double timestampPeriod = properties.limits.timestampPeriod;
    return (double)((unsigned int)timestamps[1] - (unsigned int)timestamps[0]) * timestampPeriod / 1000000.0;
}