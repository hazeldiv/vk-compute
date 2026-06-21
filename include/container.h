#ifndef container_h
#define container_h
#include <vulkan/vulkan.h>
#include "buffer.h"
#include "command.h"
#include "descriptor.h"
#include "device.h"
#include "pipeline.h"

typedef struct container {
    buffer buffer;
    command command;
    descriptor descriptor;
    device device;
    pipeline pipeline;
} container;

#endif