#ifndef device_h
#define device_h
#include <vulkan/vulkan.h>

typedef struct device {
    VkInstance instance;
    VkDevice device;
    VkQueue queue;
    VkPhysicalDevice physicalDevice;
} device;

device createDevice();
void destroyDevice(device dev);

#endif