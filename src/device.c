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
