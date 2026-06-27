#ifndef command_h
#define command_h
#include <vulkan/vulkan.h>
#include "descriptor.h"
#include "pipeline.h"

#define TIMESTAMP_QUERY_COUNT 2

typedef struct command {
    VkCommandPool pool;
    VkCommandBuffer buffer;
    VkQueryPool queryPool;
} command;

command createCommand(VkDevice device);
void destroyCommand(VkDevice device, command cmd);

#endif