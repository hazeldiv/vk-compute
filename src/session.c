#include <vulkan/vulkan.h>
#include "session.h"
#include "device.h"
#include <stdio.h>
#include <stdlib.h>

session createSession() {
    session s;
    s.dev = createDevice();

    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = 0;

    if (vkCreateCommandPool(s.dev.device, &poolInfo, NULL, &s.pool) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create command pool\n");
        exit(EXIT_FAILURE);
    }

    VkCommandBufferAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = s.pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(s.dev.device, &allocInfo, &s.buffer) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to allocate command buffer\n");
        exit(EXIT_FAILURE);
    }

    VkQueryPoolCreateInfo qpoolInfo = {0};
    qpoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpoolInfo.queryCount = TIMESTAMP_QUERY_COUNT;

    if (vkCreateQueryPool(s.dev.device, &qpoolInfo, NULL, &s.qpool) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create query pool\n");
        exit(EXIT_FAILURE);
    }

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(s.dev.device, &fenceInfo, NULL, &s.fence) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create fence\n");
        exit(EXIT_FAILURE);
    }

    return s;
}

void destroySession(session s) {
    vkDestroyFence(s.dev.device, s.fence, NULL);
    vkDestroyQueryPool(s.dev.device, s.qpool, NULL);
    vkDestroyCommandPool(s.dev.device, s.pool, NULL);
    destroyDevice(s.dev);
}