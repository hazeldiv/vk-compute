#include <vulkan/vulkan.h>
#include "device.h"
#include "pipeline.h"
#include <stdio.h>
#include <stdlib.h>

uint32_t* readShaderFile(const char filename[], size_t* outSize) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open shader file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    uint32_t* buffer = (uint32_t*)malloc(fileSize);
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation failed for shader buffer.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fread(buffer, 1, fileSize, file);
    fclose(file);
    *outSize = fileSize;
    return buffer;
}

pipeline createPipeline(VkDevice device, VkDescriptorSetLayout descriptorLayout, const char shaderPath[]) {
    pipeline pipeline = {0};
    size_t shaderSize = 0;
    uint32_t* shaderCode = readShaderFile(shaderPath, &shaderSize);

    VkShaderModuleCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = shaderSize;
    createInfo.pCode = shaderCode;

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, NULL, &shaderModule) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create shader module for %s\n", shaderPath);
        free(shaderCode);
        exit(EXIT_FAILURE);
    }
    free(shaderCode);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = NULL;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipeline.layout) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to create pipeline layout.\n");
        vkDestroyShaderModule(device, shaderModule, NULL);
        exit(EXIT_FAILURE);
    }

    VkComputePipelineCreateInfo pipelineInfo = {0};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipeline.layout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.stage.pSpecializationInfo = NULL;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline.pipeline) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to compile compute pipeline for %s\n", shaderPath);
        vkDestroyPipelineLayout(device, pipeline.layout, NULL);
        vkDestroyShaderModule(device, shaderModule, NULL);
        exit(EXIT_FAILURE);
    }
    vkDestroyShaderModule(device, shaderModule, NULL);

    return pipeline;
}

void destroyPipeline(VkDevice device, pipeline pipe) {
    vkDestroyPipeline(device, pipe.pipeline, NULL);
    vkDestroyPipelineLayout(device, pipe.layout, NULL);
}