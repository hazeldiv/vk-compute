#ifndef runner_h
#define runner_h

#include "session.h"
#include "model.h"
#include "weights.h"
#include "state.h"

typedef struct runner {
    session s;
    model_weights w;
    model_state st;
    int maxM;
} runner;

runner createRunner(session s, int maxM);
void destroyRunner(runner* r);
uint32_t runPrefill(runner* r, const uint32_t* tokens, int nTokens);
uint32_t runDecode(runner* r, uint32_t token);
void runGenerate(runner* r, const uint32_t* prompt, int nPrompt, int maxNewTokens);

#endif