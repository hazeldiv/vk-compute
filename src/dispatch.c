#include <vulkan/vulkan.h>
#include "descriptor.h"
#include "pipeline.h"
#include "command.h"
#include "dispatch.h"

void dispatch(descriptor descriptor, pipeline pipeline, command command, int x, int y, int z, int varCount, int var[]) {
    vkCmdBindPipeline(command.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);

    vkCmdBindDescriptorSets(
        command.buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout,
        0, 1, &descriptor.set,
        0, NULL
    );

    if (varCount > 0) {
        vkCmdPushConstants(
            command.buffer,
            pipeline.layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(int) * varCount,
            var
        );
    }

    vkCmdResetQueryPool(command.buffer, command.queryPool, 0, TIMESTAMP_QUERY_COUNT);

    vkCmdWriteTimestamp(command.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, command.queryPool, 0);

    vkCmdDispatch(command.buffer, (uint32_t)x, (uint32_t)y, (uint32_t)z);

    vkCmdWriteTimestamp(command.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, command.queryPool, 1);
}

void startDispatch(command command) {
    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command.buffer, &beginInfo);
}

void endDispatch(command command) {
    vkEndCommandBuffer(command.buffer);
}

dispatchContainer createDispatchContainer(device dev, int bufferCount, buffer buffers[], int varCount, char shader[]) {
    dispatchContainer VkContainer = {0};
    VkContainer.device = dev;
    VkContainer.descriptor = createDescriptor(dev.device, bufferCount, buffers);
    VkContainer.pipeline = createPipeline(dev.device, VkContainer.descriptor.layout, shader, sizeof(int) * varCount);
    VkContainer.command = createCommand(dev.device);

    return VkContainer;
}