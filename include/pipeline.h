#ifndef pipeline_h
#define pipeline_h
#include <vulkan/vulkan.h>

typedef struct pipeline {
    VkPipelineLayout layout;
    VkPipeline pipeline;
} pipeline;

void destroyPipeline(VkDevice device, pipeline pipe);

#endif