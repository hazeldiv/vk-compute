#ifndef state_h
#define state_h

#include "session.h"
#include "buffer.h"
#include "model.h"

typedef struct {
    float temperature;
    float repPenalty;
    uint32_t penaltyLength;
    uint32_t topK;
    float topP;
} sample_params;

typedef struct model_state {
    int maxM;
    buffer h;
    buffer act;
    buffer embOut;
    buffer embStaged;
    buffer yGated;
    buffer attnOut;
    buffer gAttn;
    buffer qOut;
    buffer qProj;
    buffer kProj;
    buffer vProj;
    buffer zProj;
    buffer aProj;
    buffer bProj;
    buffer kCache[MODEL_LAYERS];
    buffer vCache[MODEL_LAYERS];
    buffer kScale[MODEL_LAYERS];
    buffer kZero[MODEL_LAYERS];
    buffer vScale[MODEL_LAYERS];
    buffer vZero[MODEL_LAYERS];
    buffer stateS[MODEL_LAYERS];
    buffer convHist[MODEL_LAYERS];
    buffer position;
    buffer tokenIds;
    buffer maxValue;
    buffer maxIndex;
    buffer result;
    buffer lastRow;
    buffer gemvPartial;
    buffer qkvPartial;
    buffer ffnPartial;
    buffer linprojPartial;
    buffer attPartial;
    buffer invRms;
    buffer attScores;
    buffer smSum;
    buffer qkvRaw;
    buffer gAct;
    buffer uAct;
    buffer logits;
    buffer sampleParams;
    buffer sampleHistory;
    buffer sampleRng;
} model_state;

model_state createState(session s, const model_config* spec, int maxM, int vocab, int verbose);
void destroyState(session s, model_state* st);
void stateSetPosition(session s, model_state* st, uint32_t pos);
uint32_t stateReadPosition(session s, model_state* st);

#endif
