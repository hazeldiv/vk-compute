#ifndef state_h
#define state_h

#include "session.h"
#include "buffer.h"
#include "model.h"

typedef struct model_state {
    int maxM;
    buffer h;
    buffer act;
    buffer embOut;
    buffer yGated;
    buffer attnOut;
    buffer qOut;
    buffer qProj;
    buffer kProj;
    buffer vProj;
    buffer gProj;
    buffer aProj;
    buffer bProj;
    buffer kCache[MODEL_LAYERS];
    buffer vCache[MODEL_LAYERS];
    buffer kScale[MODEL_LAYERS];
    buffer kZero[MODEL_LAYERS];
    buffer vScale[MODEL_LAYERS];
    buffer vZero[MODEL_LAYERS];
    buffer stateS[MODEL_LAYERS];
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
} model_state;

model_state createState(session s, const model_config* spec, int maxM);
void destroyState(session s, model_state* st);
void stateSetPosition(session s, model_state* st, uint32_t pos);
uint32_t stateReadPosition(session s, model_state* st);

#endif
