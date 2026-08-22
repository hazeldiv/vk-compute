#ifndef generate_h
#define generate_h

#include "session.h"
#include "model.h"
#include "weights.h"
#include "state.h"
#include "dispatch.h"

typedef struct generator {
    session s;
    model_weights w;
    model_state st;
    const model_config* spec;
    int maxM;
    operation decodeOps[MODEL_MAX_OPS];
    int decodeOpCount;
    operation prefillOps[MODEL_MAX_OPS];
    int prefillOpCount;
    operation finalOps[MODEL_MAX_OPS];
    int finalOpCount;
} generator;

generator createGenerator(session s, const model_config* spec, int maxM);
void destroyGenerator(generator* g);
uint32_t runPrefill(generator* g, const uint32_t* tokens, int nTokens);
uint32_t runDecode(generator* g, uint32_t token);
void runGenerate(generator* g, const uint32_t* prompt, int nPrompt, int maxNewTokens);

#endif
