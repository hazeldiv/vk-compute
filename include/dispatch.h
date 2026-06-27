#ifndef dispatch_h
#define dispatch_h
#include <vulkan/vulkan.h>
#include "buffer.h"
#include "command.h"
#include "descriptor.h"
#include "device.h"
#include "pipeline.h"

typedef struct dispatchContainer {
    buffer buffer;
    command command;
    descriptor descriptor;
    device device;
    pipeline pipeline;
} dispatchContainer;

void startDispatch(command command);
void endDispatch(command command);
void dispatch(descriptor descriptor, pipeline pipeline, command command, int x, int y, int z, int varCount, int var[]);
dispatchContainer createDispatchContainer(device dev, int bufferCount, buffer buffers[], int varCount, char shader[]);

#endif