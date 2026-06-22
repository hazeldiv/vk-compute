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
void startDispatch(command command);
void endDispatch(command command);
void dispatch(descriptor descriptor, pipeline pipeline, command command, int x, int y, int z, int varCount, int var[varCount]);

#endif