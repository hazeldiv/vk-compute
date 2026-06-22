#ifndef buffer_h
#define buffer_h
#include <vulkan/vulkan.h>

typedef struct buffer {
    VkBuffer buffer;
    VkMemoryRequirements memReqs;
    VkDeviceMemory memory;
    void* mappedMemory;
    void* hostMemory;
    int64_t size;
} buffer;

buffer createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, void* hostMemory, int64_t size);
void destroyBuffer(VkDevice device, buffer buf);

#endif