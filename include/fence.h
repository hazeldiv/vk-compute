#ifndef fence_h
#define fence_h
#include <vulkan/vulkan.h>
#include "device.h"
#include "command.h"

VkFence createFence(device dev, command command);

#endif