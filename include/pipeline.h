#ifndef pipeline_h
#define pipeline_h
#include <vulkan/vulkan.h>

#define MAX_PUSH_CONSTANT_SIZE 128

typedef struct pipeline {
    VkPipelineLayout layout;
    VkPipeline pipeline;
} pipeline;

pipeline createPipeline(VkDevice device, VkDescriptorSetLayout descriptorLayout, const char shaderPath[], uint32_t pushConstantSize);
void destroyPipeline(VkDevice device, pipeline pipe);

#endif