#include <stdio.h>
#include <stdlib.h>
#include "session.h"
#include "runner.h"

void runDemo() {
    session s = createSession();
    runner r = createRunner(s, MODEL_MAX_GEMM);

    uint32_t prompt[MODEL_MAX_GEMM];
    for (int i = 0; i < MODEL_MAX_GEMM; i++) {
        prompt[i] = (uint32_t)((i * 1237 + 555) % MODEL_VOCAB);
    }

    runGenerate(&r, prompt, MODEL_MAX_GEMM, 8);

    destroyRunner(&r);
    destroySession(s);
}