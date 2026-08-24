#ifndef dispatch_h
#define dispatch_h

#include <vulkan/vulkan.h>
#include "buffer.h"
#include "device.h"
#include "descriptor.h"
#include "pipeline.h"
#include "session.h"

#define MAX_OP_BUFFERS 16
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
    int layer;
} operation;

void execute(session s, operation ops[], int opCount);
void executeLogged(session s, operation ops[], int opCount, const char* phase, int token);
void setTimingEnabled(int enabled);
void closeTimingLog(void);
double getExecutionTime(session s);

#endif