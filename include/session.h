#ifndef session_h
#define session_h

#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"

#define TIMESTAMP_QUERY_COUNT 2

typedef struct session {
    device dev;
    VkCommandPool pool;
    VkCommandBuffer buffer;
    VkFence fence;
    VkQueryPool qpool;
} session;

session createSession();
void destroySession(session s);

#endif