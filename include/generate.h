#ifndef generate_h
#define generate_h

#include "session.h"
#include "model.h"
#include "weights.h"
#include "state.h"
#include "dispatch.h"

#define DECODE_GROUP 4

typedef struct generator {
    session s;
    model_weights w;
    model_state st;
    const model_config* spec;
    int maxM;
    operation groupOps[MODEL_MAX_OPS];
    int groupOpCount;
    operation prefillOps[MODEL_MAX_OPS];
    int prefillOpCount;
    operation finalOps[MODEL_MAX_OPS];
    int finalOpCount;
    uint32_t nextPos;
} generator;

generator* createGenerator(session s, const model_config* spec, int maxM);
void destroyGenerator(generator* g);
uint32_t runPrefill(generator* g, const uint32_t* tokens, int nTokens);
void runGenerate(generator* g, const uint32_t* prompt, int nPrompt, int maxNewTokens);

#endif
