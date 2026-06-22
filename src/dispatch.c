#include <vulkan/vulkan.h>
#include "descriptor.h"
#include "pipeline.h"
#include "command.h"

void dispatch(descriptor descriptor, pipeline pipeline, command command, int x, int y, int z, int varCount, int var[varCount]) {
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

    vkCmdDispatch(command.buffer, (uint32_t)x, (uint32_t)y, (uint32_t)z);
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