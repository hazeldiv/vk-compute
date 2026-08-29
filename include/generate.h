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
    operation groupOpsShort[MODEL_MAX_OPS];
    int groupOpCountShort;
    operation prefillOps[MODEL_MAX_OPS];
    int prefillOpCount;
    operation finalOps[MODEL_MAX_OPS];
    int finalOpCount;
    uint32_t nextPos;
    int vocab;
    int eos;
    int maxCtx;
    char dumpDir[256];
    int dumpLayers;
} generator;

generator* createGenerator(session s, const model_config* spec, int maxM, const char* weightDir, const char* customHead, const char* customEmbed, int eos, int maxCtx);
void destroyGenerator(generator* g);
uint32_t runPrefill(generator* g, const uint32_t* tokens, int nTokens);
uint32_t generateTokens(generator* g, const uint32_t* prompt, int nPrompt, int maxNewTokens, uint32_t* out, int* outCount);
void resetGenerator(generator* g);
void generatorSetDumpDir(generator* g, const char* dir);
void generatorDumpPrefill(generator* g, int rows);
void generatorSetDumpLayers(generator* g, int layers);
void generatorDumpDecodeStep(generator* g, int step);

#endif
