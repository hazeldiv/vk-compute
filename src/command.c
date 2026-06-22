#include <vulkan/vulkan.h>
#include "command.h"
#include <stdio.h>
#include <stdlib.h>

command createCommand(VkDevice device, VkPhysicalDevice physicalDevice) {
    command command = {0};

    VkCommandPoolCreateInfo poolInfo;
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = 0;

    if (vkCreateCommandPool(device, &poolInfo, NULL, &command.pool) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create command pool.\n");
        exit(EXIT_FAILURE);
    }

    VkCommandBufferAllocateInfo allocInfo;
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = command.pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &command.buffer) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to allocate command buffer.\n");
        exit(EXIT_FAILURE);
    }

    VkQueryPoolCreateInfo queryPoolInfo;
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = TIMESTAMP_QUERY_COUNT;
    queryPoolInfo.flags = 0;

    if (vkCreateQueryPool(device, &queryPoolInfo, NULL, &command.queryPool) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create query pool.\n");
        exit(EXIT_FAILURE);
    }

    return command;
}

void destroyCommand(VkDevice device, command cmd) {
    vkDestroyQueryPool(device, cmd.queryPool, NULL);
    vkDestroyCommandPool(device, cmd.pool, NULL);
}