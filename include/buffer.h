#ifndef buffer_h
#define buffer_h
#include <vulkan/vulkan.h>

#define MEMORY_RAM 0
#define MEMORY_VRAM 1

typedef struct buffer {
    VkBuffer buffer;
    VkMemoryRequirements memReqs;
    VkDeviceMemory memory;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    void* mappedMemory;
    int64_t size;
    int memoryType;
} buffer;

buffer createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, void* data, int64_t size, int memoryType);
void createTransferAndCopy(VkDevice device, VkQueue queue, buffer* buffers, int bufferCount);
void readBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, buffer buf, void* output);
void destroyBuffer(VkDevice device, buffer buf);

#endif
