#ifndef command_h
#define command_h
#include <vulkan/vulkan.h>

typedef struct command {
    VkCommandPool pool;
    VkCommandBuffer buffer;
} command;

void destroyCommand(VkDevice device, command cmd);

#endif