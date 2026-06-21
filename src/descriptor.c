#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"
#include "descriptor.h"
#include <stdio.h>
#include <stdlib.h>

descriptor createDescriptor(VkDevice device, int bufferCount, buffer buffer[bufferCount]) {
    descriptor descriptor = {0};
    VkDescriptorSetLayoutBinding bindings[bufferCount];
    
    for (int i = 0; i < bufferCount; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = NULL;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bufferCount;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptor.layout) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create descriptor set layout!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorPoolSize poolSize = {0};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bufferCount;

    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptor.pool) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create descriptor pool!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptor.pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptor.layout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptor.set) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to allocate descriptor set!\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorBufferInfo bufferInfos[bufferCount];
    VkWriteDescriptorSet descriptorWrites[bufferCount];

    for (int i = 0; i < bufferCount; i++) {
        bufferInfos[i].buffer = buffer[i].buffer;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = NULL;
        descriptorWrites[i].dstSet = descriptor.set;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pImageInfo = NULL;
        descriptorWrites[i].pTexelBufferView = NULL;
    }

    vkUpdateDescriptorSets(device, bufferCount, descriptorWrites, 0, NULL);

    return descriptor;
}

void destroyDescriptor(VkDevice device, descriptor desc) {
    vkDestroyDescriptorPool(device, desc.pool, NULL);
    vkDestroyDescriptorSetLayout(device, desc.layout, NULL);
}