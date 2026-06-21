#ifndef descriptor_h
#define descriptor_h
#include <vulkan/vulkan.h>

typedef struct descriptor {
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
} descriptor;

void destroyDescriptor(VkDevice device, descriptor desc);

#endif