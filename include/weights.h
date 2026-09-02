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
    buffer* gammaIn;
    buffer* gammaF;
    buffer* qNorm;
    buffer* kNorm;
    tensor* proj;
    tensor* out;
    tensor* gate;
    tensor* up;
    tensor* down;
    buffer* conv;
    buffer* aLog;
    buffer* dtBias;
    buffer* attnNorm;
    buffer* layerBufs;
    tensor* tensorBufs;
    int vocab;
    int layerCount;
} model_weights;

model_weights createWeights(session s, const model_config* spec, const char* weightDir, int verbose);
void destroyWeights(session s, model_weights* w);

#endif
