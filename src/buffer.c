#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

buffer createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, void* hostMemory, int64_t size) {
    buffer buffer = {0};
    VkBufferCreateInfo bufferInfo = {0};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &bufferInfo, NULL, &buffer.buffer) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create buffer handle!\n");
        exit(EXIT_FAILURE);
    }

    vkGetBufferMemoryRequirements(device, buffer.buffer, &buffer.memReqs);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    uint32_t memoryTypeIndex = UINT32_MAX;
    VkMemoryPropertyFlags requiredProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((buffer.memReqs.memoryTypeBits & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties) {
            memoryTypeIndex = i;
            break;
        }
    }
    
    if (memoryTypeIndex == UINT32_MAX) {
        fprintf(stderr, "Error: Failed to find a host-visible memory type for this buffer!\n");
        exit(EXIT_FAILURE);
    }
    
    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = buffer.memReqs.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(device, &allocInfo, NULL, &buffer.memory) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to allocate GPU memory!\n");
        exit(EXIT_FAILURE);
    }
    vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0);

    void* mappedMemory;
    vkMapMemory(device, buffer.memory, 0, size, 0, &mappedMemory);
    memcpy(mappedMemory, hostMemory, size);
    buffer.mappedMemory = mappedMemory;
    buffer.hostMemory = hostMemory;
    buffer.size = size;
    return buffer;
}

void destroyBuffer(VkDevice device, buffer buf) {
    vkUnmapMemory(device, buf.memory);
    vkDestroyBuffer(device, buf.buffer, NULL);
    vkFreeMemory(device, buf.memory, NULL);
}