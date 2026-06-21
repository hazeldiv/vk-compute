#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"
#include "descriptor.h"
#include "pipeline.h"
#include "command.h"
#include "container.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

device createDevice();
buffer createBuffer();
descriptor createDescriptor();
pipeline createPipeline();
command createCommand();
VkFence createFence();
void dispatch();
void startDispatch();
void endDispatch();

void cleanup(device dev, buffer buf, descriptor desc, pipeline pipe, command cmd, VkFence fence);

container createVKContainer(device dev, int bufferCount, buffer buffer[bufferCount]) {
    container VkContainer = {0};
    VkContainer.device = dev;
    VkContainer.descriptor = createDescriptor(dev.device, bufferCount, buffer);
    VkContainer.pipeline = createPipeline(dev.device, VkContainer.descriptor.layout, "shader.spv");
    VkContainer.command = createCommand(VkContainer.device.device);
    return VkContainer;
}

void compute() {
    device dev = createDevice();

    float hostData[256];
    for (int i=0;i<256;i++) {
        hostData[i] = (i%5 + 1) * 0.1f;
    }
    buffer bufferA = createBuffer(dev.device, dev.physicalDevice, hostData, sizeof(hostData));
    container VkContainer = createVKContainer(dev, 1, &bufferA);

    startDispatch(VkContainer.command);
    dispatch(VkContainer.descriptor, VkContainer.pipeline, VkContainer.command, 1,1,1);
    endDispatch(VkContainer.command);

    VkFence fence = createFence(dev, VkContainer.command);
    float* outputResults = (float*)bufferA.mappedMemory;
    printf("Results from C Vulkan Dispatch: ");
    for (int i=0;i<256;i++) {
        printf("%f ", outputResults[i]);
    }
    printf("\n");

    cleanup(dev, bufferA, VkContainer.descriptor, VkContainer.pipeline, VkContainer.command, fence);
}

void cleanup(device dev, buffer buf, descriptor desc, pipeline pipe, command cmd, VkFence fence) {
    vkDestroyFence(dev.device, fence, NULL);
    destroyCommand(dev.device, cmd);
    destroyPipeline(dev.device, pipe);
    destroyDescriptor(dev.device, desc);
    destroyBuffer(dev.device, buf);
    destroyDevice(dev);
}
