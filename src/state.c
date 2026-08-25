#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "state.h"

static buffer createZeroed(session s, int64_t size) {
    uint8_t* zeros = (uint8_t*)calloc(1, (size_t)size);
    buffer b = createBuffer(s.dev.device, s.dev.physicalDevice, zeros, size, MEMORY_VRAM);
    free(zeros);
    return b;
}

model_state createState(session s, const model_config* spec, int maxM) {
    model_state st = {0};
    st.maxM = maxM;
    int mn = maxM * MODEL_K;

    st.h = createZeroed(s, (int64_t)sizeof(float) * mn);
    st.act = createZeroed(s, (int64_t)sizeof(float) * maxM * MODEL_FFN_N);
    st.embOut = createZeroed(s, (int64_t)sizeof(float) * mn);
    st.yGated = createZeroed(s, (int64_t)sizeof(float) * mn);
    st.attnOut = createZeroed(s, (int64_t)sizeof(float) * mn);
    st.qOut = createZeroed(s, (int64_t)sizeof(float) * mn);
    st.qProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (MODEL_N_QK * MODEL_DIM));
    st.kProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (MODEL_N_QK * MODEL_DIM));
    st.vProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (MODEL_N_V * MODEL_DIM));
    st.gProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (MODEL_N_V * MODEL_DIM));
    st.aProj = createZeroed(s, (int64_t)sizeof(float) * maxM * MODEL_N_QK);
    st.bProj = createZeroed(s, (int64_t)sizeof(float) * maxM * MODEL_N_QK);

    for (int L = 0; L < spec->layerCount; L++) {
        const layer* ly = &spec->layers[L];
if (ly->attn.type == ATTENTION_FULL) {
            int64_t cacheBytes = (int64_t)MODEL_KV_ROWS * MODEL_MAX_CTX;
            if (ly->attn.q != QUANT_FP16) {
                st.kCache[L] = createZeroed(s, cacheBytes);
                st.vCache[L] = createZeroed(s, cacheBytes);
                st.kScale[L] = createZeroed(s, (int64_t)sizeof(float) * MODEL_KV_HEADS * MODEL_MAX_CTX);
                st.kZero[L] = createZeroed(s, (int64_t)sizeof(float) * MODEL_KV_HEADS * MODEL_MAX_CTX);
                st.vScale[L] = createZeroed(s, (int64_t)sizeof(float) * MODEL_KV_HEADS * MODEL_MAX_CTX);
                st.vZero[L] = createZeroed(s, (int64_t)sizeof(float) * MODEL_KV_HEADS * MODEL_MAX_CTX);
                printf("Allocating %lld MB for layer %d KV cache\n", (long long)cacheBytes * 2 / (1024 * 1024), L);
            } else {
                printf("Allocating %lld MB for layer %d KV cache\n", (long long)cacheBytes * 4 / (1024 * 1024), L);
                st.kCache[L] = createZeroed(s, cacheBytes * 2);
                st.vCache[L] = createZeroed(s, cacheBytes * 2);
            }
        } else if (ly->attn.type == ATTENTION_DELTA) {
            st.stateS[L] = createZeroed(s, (int64_t)sizeof(float) * MODEL_N_V * MODEL_DIM * MODEL_DIM);
        }
    }

    uint32_t zero = 0;
    st.position = createBuffer(s.dev.device, s.dev.physicalDevice, &zero, sizeof(uint32_t), MEMORY_RAM);
    uint32_t* tokenInit = (uint32_t*)calloc(maxM, sizeof(uint32_t));
    st.tokenIds = createBuffer(s.dev.device, s.dev.physicalDevice, tokenInit, (int64_t)sizeof(uint32_t) * maxM, MEMORY_RAM);
    free(tokenInit);
    float* lastInit = (float*)calloc(MODEL_K, sizeof(float));
    st.lastRow = createBuffer(s.dev.device, s.dev.physicalDevice, lastInit, (int64_t)sizeof(float) * MODEL_K, MEMORY_RAM);
    free(lastInit);
    int numGroups = (MODEL_VOCAB + 255) / 256;
    st.maxValue = createZeroed(s, (int64_t)sizeof(float) * numGroups);
    st.maxIndex = createZeroed(s, (int64_t)sizeof(uint32_t) * numGroups);
    st.result = createZeroed(s, sizeof(uint32_t) * 4);
    st.gemvPartial = createZeroed(s, (int64_t)sizeof(float) * 4 * MODEL_K);
    st.qkvPartial = createZeroed(s, (int64_t)sizeof(float) * 4 * MODEL_QKV_N);
    st.ffnPartial = createZeroed(s, (int64_t)sizeof(float) * 8 * MODEL_FFN_N);
    st.linprojPartial = createZeroed(s, (int64_t)sizeof(float) * 4 * MODEL_PROJ_N);
    st.attPartial = createZeroed(s, (int64_t)sizeof(float) * 528384);
    st.invRms = createZeroed(s, (int64_t)sizeof(float) * maxM);
    st.attScores = createZeroed(s, (int64_t)maxM * MODEL_MAX_CTX * 4 * 2);
    st.gAct = createZeroed(s, (int64_t)sizeof(float) * maxM * MODEL_FFN_N);
    st.uAct = createZeroed(s, (int64_t)sizeof(float) * maxM * MODEL_FFN_N);

    buffer bufs[] = {
        st.h, st.act, st.embOut, st.yGated, st.attnOut, st.qOut,
        st.qProj, st.kProj, st.vProj, st.gProj, st.aProj, st.bProj,
        st.maxValue, st.maxIndex, st.result
    };
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 15);
    for (int L = 0; L < MODEL_LAYERS; L++) {
        if (st.kCache[L].buffer != VK_NULL_HANDLE) {
            buffer ks[] = {st.kCache[L], st.vCache[L], st.kScale[L], st.kZero[L], st.vScale[L], st.vZero[L]};
            createTransferAndCopy(s.dev.device, s.dev.queue, ks, 6);
        }
        if (st.stateS[L].buffer != VK_NULL_HANDLE) {
            buffer ss[] = {st.stateS[L]};
            createTransferAndCopy(s.dev.device, s.dev.queue, ss, 1);
        }
    }

    return st;
}

void destroyState(session s, model_state* st) {
    destroyBuffer(s.dev.device, st->h);
    destroyBuffer(s.dev.device, st->act);
    destroyBuffer(s.dev.device, st->embOut);
    destroyBuffer(s.dev.device, st->yGated);
    destroyBuffer(s.dev.device, st->attnOut);
    destroyBuffer(s.dev.device, st->qOut);
    destroyBuffer(s.dev.device, st->qProj);
    destroyBuffer(s.dev.device, st->kProj);
    destroyBuffer(s.dev.device, st->vProj);
    destroyBuffer(s.dev.device, st->gProj);
    destroyBuffer(s.dev.device, st->aProj);
    destroyBuffer(s.dev.device, st->bProj);
    for (int L = 0; L < MODEL_LAYERS; L++) {
        if (st->kCache[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->kCache[L]);
        if (st->vCache[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->vCache[L]);
        if (st->kScale[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->kScale[L]);
        if (st->kZero[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->kZero[L]);
        if (st->vScale[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->vScale[L]);
        if (st->vZero[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->vZero[L]);
        if (st->stateS[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->stateS[L]);
    }
    destroyBuffer(s.dev.device, st->position);
    destroyBuffer(s.dev.device, st->tokenIds);
    destroyBuffer(s.dev.device, st->maxValue);
    destroyBuffer(s.dev.device, st->maxIndex);
    destroyBuffer(s.dev.device, st->result);
    destroyBuffer(s.dev.device, st->lastRow);
    destroyBuffer(s.dev.device, st->gemvPartial);
    destroyBuffer(s.dev.device, st->qkvPartial);
    destroyBuffer(s.dev.device, st->ffnPartial);
    destroyBuffer(s.dev.device, st->linprojPartial);
    destroyBuffer(s.dev.device, st->attPartial);
    destroyBuffer(s.dev.device, st->invRms);
    destroyBuffer(s.dev.device, st->attScores);
    destroyBuffer(s.dev.device, st->gAct);
    destroyBuffer(s.dev.device, st->uAct);
}

void stateSetPosition(session s, model_state* st, uint32_t pos) {
    (void)s;
    if (st->position.mappedMemory != NULL) {
        *(uint32_t*)st->position.mappedMemory = pos;
    }
}

uint32_t stateReadPosition(session s, model_state* st) {
    uint32_t pos = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, st->position, &pos);
    return pos;
}
