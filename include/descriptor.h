#ifndef descriptor_h
#define descriptor_h
#include <vulkan/vulkan.h>
#include "buffer.h"

typedef struct descriptor {
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
} descriptor;

descriptor createDescriptor(VkDevice device, int bufferCount, buffer buffer[bufferCount]);
void destroyDescriptor(VkDevice device, descriptor desc);

#endif