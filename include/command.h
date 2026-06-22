#ifndef command_h
#define command_h
#include <vulkan/vulkan.h>

#define TIMESTAMP_QUERY_COUNT 2

typedef struct command {
    VkCommandPool pool;
    VkCommandBuffer buffer;
    VkQueryPool queryPool;
} command;

command createCommand(VkDevice device, VkPhysicalDevice physicalDevice);
void destroyCommand(VkDevice device, command cmd);
void startDispatch(command command);
void endDispatch(command command);

#endif