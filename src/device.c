#include <vulkan/vulkan.h>
#include "device.h"
#include <stdio.h>
#include <stdlib.h>

device createDevice() {
    device dev = {0};

    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Compute";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, NULL, &dev.instance) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create Vulkan instance!\n");
        exit(EXIT_FAILURE);
    }

    uint32_t deviceCount = 1;
    if (vkEnumeratePhysicalDevices(dev.instance, &deviceCount, &dev.physicalDevice) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to enumerate physical devices!\n");
        exit(EXIT_FAILURE);
    }
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No Vulkan-compatible GPU found!\n");
        exit(EXIT_FAILURE);
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {0};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {0};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    if (vkCreateDevice(dev.physicalDevice, &deviceCreateInfo, NULL, &dev.device) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create logical device!\n");
        exit(EXIT_FAILURE);
    }
    vkGetDeviceQueue(dev.device, 0, 0, &dev.queue);
    return dev;
}

void destroyDevice(device dev) {
    vkDestroyDevice(dev.device, NULL);
    vkDestroyInstance(dev.instance, NULL);
}

void dumpMemoryInfo(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);

    fprintf(stderr, "== memory heaps ==\n");
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
        fprintf(stderr, "  heap[%u]: %.0f MB  %s\n", i,
                (double)mp.memoryHeaps[i].size / (1024.0 * 1024.0),
                (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL" : "system");
    }

    fprintf(stderr, "== memory types ==\n");
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        fprintf(stderr, "  type[%u] -> heap[%u] : %s%s%s%s\n", i, mp.memoryTypes[i].heapIndex,
                (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
                (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
                (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED " : "");
    }
}
