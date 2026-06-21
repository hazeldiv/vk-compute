#include "command.h"
#include "device.h"
#include <stdio.h>
#include <stdlib.h>

VkFence createFence(device dev, command command) {
    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence;
    if (vkCreateFence(dev.device, &fenceInfo, NULL, &fence) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create fence!\n");
        exit(EXIT_FAILURE);
    }

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command.buffer;
    vkQueueSubmit(dev.queue, 1, &submitInfo, fence);
    vkWaitForFences(dev.device, 1, &fence, VK_TRUE, UINT64_MAX);
    return fence;
}