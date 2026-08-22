#ifndef descriptor_h
#define descriptor_h
#include <vulkan/vulkan.h>
#include "buffer.h"

typedef struct descriptor {
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
} descriptor;

descriptor createDescriptor(VkDevice device, int bufferCount, buffer buffers[]);
void destroyDescriptor(VkDevice device, descriptor desc);
VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device, int bufferCount);
descriptor createDescriptorSetFromLayout(VkDevice device, VkDescriptorSetLayout layout, int bufferCount, buffer buffers[]);
void destroyDescriptorSet(VkDevice device, descriptor desc);

#endif