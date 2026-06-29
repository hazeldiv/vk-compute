#ifndef dispatch_h
#define dispatch_h

#include <vulkan/vulkan.h>
#include "buffer.h"
#include "device.h"
#include "descriptor.h"
#include "pipeline.h"
#include "session.h"

#define MAX_OP_BUFFERS 8
#define MAX_PUSH_CONSTANTS 8

typedef struct operation {
    char shader[128];
    buffer buffers[MAX_OP_BUFFERS];
    int bufferCount;
    int pushConstants[MAX_PUSH_CONSTANTS];
    int pushConstantCount;
    int dispatchX;
    int dispatchY;
    int dispatchZ;
} operation;

void execute(session s, operation ops[], int opCount);
double getExecutionTime(session s);

#endif