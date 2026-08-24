#ifndef session_h
#define session_h

#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"

#define TIMESTAMP_QUERY_COUNT 8192
#define FRAME_COUNT 3

typedef struct session {
    device dev;
    VkCommandPool pool;
    VkCommandBuffer buffer[FRAME_COUNT];
    VkFence fence[FRAME_COUNT];
    VkQueryPool qpool;
    uint32_t frame;
    int lastSubmitted;
} session;

session createSession();
void destroySession(session s);

#endif