#ifndef weights_h
#define weights_h

#include "session.h"
#include "buffer.h"
#include "data.h"
#include "model.h"
#include "safetensors.h"

typedef struct tensor {
    buffer data;
    buffer scale;
    buffer zero;
    QuantType q;
    int rows;
    int cols;
} tensor;

typedef struct model_weights {
    buffer theta;
    buffer embed;
    buffer lmHead;
    buffer gammaFinal;
    buffer gammaIn[MODEL_LAYERS];
    buffer gammaF[MODEL_LAYERS];
    buffer qNorm[MODEL_LAYERS];
    buffer kNorm[MODEL_LAYERS];
    tensor proj[MODEL_LAYERS];
    tensor out[MODEL_LAYERS];
    tensor gate[MODEL_LAYERS];
    tensor up[MODEL_LAYERS];
    tensor down[MODEL_LAYERS];
    buffer conv[MODEL_LAYERS];
    buffer aLog[MODEL_LAYERS];
    buffer dtBias[MODEL_LAYERS];
    buffer attnNorm[MODEL_LAYERS];
    int vocab;
} model_weights;

model_weights createWeights(session s, const model_config* spec, const char* weightDir, const char* customHead, const char* customEmbed, int verbose);
void destroyWeights(session s, model_weights* w);

#endif
