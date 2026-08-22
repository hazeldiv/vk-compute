#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runner.h"
#include "dispatch.h"

static void addOp(operation* ops, int* n, const char* shader, buffer* bufs, int bc, const int* push, int pc, int dx, int dy) {
    operation* op = &ops[*n];
    memset(op, 0, sizeof(operation));
    snprintf(op->shader, sizeof(op->shader), "%s", shader);
    memcpy(op->buffers, bufs, sizeof(buffer) * bc);
    op->bufferCount = bc;
    memcpy(op->pushConstants, push, sizeof(int) * pc);
    op->pushConstantCount = pc;
    op->dispatchX = dx;
    op->dispatchY = dy;
    op->dispatchZ = 1;
    (*n)++;
}

static void addGemmAdd(runner* r, operation* ops, int* n, const tensor* wt, buffer input, buffer output, buffer residual, QuantType q, int m) {
    buffer bufs[6];
    int b = 0;
    bufs[b++] = input;
    bufs[b++] = wt->data;
    bufs[b++] = output;
    if (q != QUANT_FP16) {
        bufs[b++] = wt->scale;
        bufs[b++] = wt->zero;
    }
    bufs[b++] = residual;
    int k = (input.buffer == r->st.act.buffer) ? MODEL_FFN_N : MODEL_K;
    int push[] = {m, MODEL_K, k};
    addOp(ops, n, model_shader(m == 1 ? "GEMV-ADD" : "GEMM-ADD", q), bufs, b, push, 3,
          m == 1 ? MODEL_K / 256 : MODEL_K / 16, m == 1 ? 1 : m / 16);
}

static void addLinearProj(runner* r, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &r->st;
    model_weights* w = &r->w;
    QuantType q = model_layer_q[L];
    buffer bufs[14];
    int b = 0;
    bufs[b++] = st->h;
    bufs[b++] = w->gammaIn[L];
    bufs[b++] = w->proj[L].data;
    if (!gemm && q != QUANT_FP16) {
        bufs[b++] = w->proj[L].scale;
        bufs[b++] = w->proj[L].zero;
    }
    bufs[b++] = st->qProj;
    bufs[b++] = st->kProj;
    bufs[b++] = st->vProj;
    bufs[b++] = st->gProj;
    bufs[b++] = st->aProj;
    bufs[b++] = st->bProj;
    if (gemm && q != QUANT_FP16) {
        bufs[b++] = w->proj[L].scale;
        bufs[b++] = w->proj[L].zero;
    }
    int push[] = {m, MODEL_PROJ_N, MODEL_K};
    addOp(ops, n, model_shader(gemm ? "RmsNorm-LinearProj-GEMM" : "RmsNorm-LinearProj", q), bufs, b, push, 3,
          gemm ? MODEL_PROJ_N / 16 : (MODEL_PROJ_N + 255) / 256, gemm ? m / 16 : 1);
}

static void buildFfn(runner* r, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &r->st;
    model_weights* w = &r->w;
    QuantType f = model_ffn_q[L];
    buffer ffnBufs[11];
    int bf = 0;
    ffnBufs[bf++] = st->h;
    ffnBufs[bf++] = w->gammaF[L];
    ffnBufs[bf++] = w->gate[L].data;
    ffnBufs[bf++] = w->up[L].data;
    ffnBufs[bf++] = st->act;
    if (f != QUANT_FP16) {
        ffnBufs[bf++] = w->gate[L].scale;
        ffnBufs[bf++] = w->gate[L].zero;
        ffnBufs[bf++] = w->up[L].scale;
        ffnBufs[bf++] = w->up[L].zero;
    }
    int push[] = {m, MODEL_FFN_N, MODEL_K};
    addOp(ops, n, model_shader(gemm ? "RmsNorm-swiglu-ffn-GEMM" : "RmsNorm-swiglu-ffn", f), ffnBufs, bf, push, 3,
          gemm ? MODEL_FFN_N / 16 : (MODEL_FFN_N + 255) / 256, gemm ? m / 16 : 1);

    addGemmAdd(r, ops, n, &w->down[L], st->act, st->h, st->h, f, m);
}

static void buildLmHead(runner* r, operation* ops, int* n, buffer* input, int doIncrement) {
    model_state* st = &r->st;
    model_weights* w = &r->w;

    buffer lmBufs[] = {*input, w->lmHead, st->maxValue, st->maxIndex, w->gammaFinal};
    int pushL[] = {1, MODEL_VOCAB, MODEL_K};
    addOp(ops, n, "LMHead-GEMV-ArgMax-FP16.spv", lmBufs, 5, pushL, 3, (MODEL_VOCAB + 255) / 256, 1);

    buffer redBufs[] = {st->maxValue, st->maxIndex, st->result, st->position};
    int pushR[] = {MODEL_VOCAB, doIncrement};
    addOp(ops, n, "ArgMax-Reduce.spv", redBufs, 4, pushR, 2, 1, 1);
}

uint32_t runPrefill(runner* r, const uint32_t* tokens, int nTokens) {
    int m = r->maxM;
    model_state* st = &r->st;
    model_weights* w = &r->w;

    memcpy(st->tokenIds.mappedMemory, tokens, sizeof(uint32_t) * r->maxM);

    operation ops[MODEL_MAX_OPS];
    int n = 0;

    buffer embedBufs[] = {st->tokenIds, w->lmHead, w->gammaIn[0], w->proj[0].data, st->qProj, st->kProj, st->vProj, st->gProj, st->aProj, st->bProj, st->embOut};
    int pushE[] = {m, MODEL_PROJ_N, MODEL_K, MODEL_VOCAB};
    addOp(ops, &n, "Embed-RmsNorm-LinearProj-GEMM-FP16.spv", embedBufs, 11, pushE, 4, MODEL_PROJ_N / 16, m / 16);

    for (int L = 0; L < MODEL_LAYERS; L++) {
        QuantType q = model_layer_q[L];
        if (model_attn_layer[L]) {
            buffer qkvBufs[14];
            int b = 0;
            qkvBufs[b++] = st->h;
            qkvBufs[b++] = w->gammaIn[L];
            qkvBufs[b++] = w->proj[L].data;
            if (q != QUANT_FP16) {
                qkvBufs[b++] = w->proj[L].scale;
                qkvBufs[b++] = w->proj[L].zero;
            }
            qkvBufs[b++] = w->theta;
            qkvBufs[b++] = st->qOut;
            qkvBufs[b++] = st->kCache[L];
            qkvBufs[b++] = st->vCache[L];
            if (q == QUANT_INT4) {
                qkvBufs[b++] = st->kScale[L];
                qkvBufs[b++] = st->kZero[L];
                qkvBufs[b++] = st->vScale[L];
                qkvBufs[b++] = st->vZero[L];
            }
            qkvBufs[b++] = st->position;
            int push[] = {m, MODEL_QKV_N, MODEL_K, 0, MODEL_Q_OFF, MODEL_V_OFF};
            addOp(ops, &n, model_shader("RmsNorm-QKV-GEMM", q), qkvBufs, b, push, 6, MODEL_HEADS + 2 * MODEL_KV_HEADS, m / 16);

            buffer attBufs[9];
            int ba = 0;
            attBufs[ba++] = st->kCache[L];
            attBufs[ba++] = st->vCache[L];
            attBufs[ba++] = st->qOut;
            attBufs[ba++] = st->attnOut;
            if (q == QUANT_INT4) {
                attBufs[ba++] = st->kScale[L];
                attBufs[ba++] = st->kZero[L];
                attBufs[ba++] = st->vScale[L];
                attBufs[ba++] = st->vZero[L];
            }
            int pushA[] = {m};
            addOp(ops, &n, model_shader("Att-full-GEMM", q), attBufs, ba, pushA, 1, MODEL_HEADS, m / 16);

            addGemmAdd(r, ops, &n, &w->out[L], st->attnOut, st->h, st->h, q, m);
        } else {
            if (L != 0) {
                addLinearProj(r, ops, &n, L, 1, m);
            }

            buffer dnBufs[] = {st->qProj, st->kProj, st->vProj, st->gProj, st->aProj, st->bProj, st->stateS[L], st->yGated};
            int pushDn[] = {m};
            addOp(ops, &n, "GatedDeltaNet-GEMM.spv", dnBufs, 8, pushDn, 1, MODEL_N_V, 1);

            addGemmAdd(r, ops, &n, &w->out[L], st->yGated, st->h, L == 0 ? st->embOut : st->h, q, m);
        }

        buildFfn(r, ops, &n, L, 1, m);
    }

    execute(r->s, ops, n);

    float* hCpu = (float*)malloc(sizeof(float) * m * MODEL_K);
    readBuffer(r->s.dev.device, r->s.dev.physicalDevice, r->s.dev.queue, st->h, hCpu);
    memcpy(st->lastRow.mappedMemory, hCpu + (nTokens - 1) * MODEL_K, sizeof(float) * MODEL_K);
    free(hCpu);

    operation finalOps[4];
    int nf = 0;
    buildLmHead(r, finalOps, &nf, &st->lastRow, 0);
    execute(r->s, finalOps, nf);

    uint32_t result = 0;
    readBuffer(r->s.dev.device, r->s.dev.physicalDevice, r->s.dev.queue, st->result, &result);
    return result;
}

uint32_t runDecode(runner* r, uint32_t token) {
    model_state* st = &r->st;
    model_weights* w = &r->w;

    ((uint32_t*)st->tokenIds.mappedMemory)[0] = token;

    operation ops[MODEL_MAX_OPS];
    int n = 0;

    buffer embedBufs[] = {st->tokenIds, w->lmHead, w->gammaIn[0], w->proj[0].data, st->qProj, st->kProj, st->vProj, st->gProj, st->aProj, st->bProj, st->embOut};
    int pushE[] = {1, MODEL_PROJ_N, MODEL_K, MODEL_VOCAB};
    addOp(ops, &n, "Embed-RmsNorm-LinearProj-FP16.spv", embedBufs, 11, pushE, 4, (MODEL_PROJ_N + 255) / 256, 1);

    for (int L = 0; L < MODEL_LAYERS; L++) {
        QuantType q = model_layer_q[L];
        if (model_attn_layer[L]) {
            buffer qkvBufs[14];
            int b = 0;
            qkvBufs[b++] = st->h;
            qkvBufs[b++] = w->gammaIn[L];
            qkvBufs[b++] = w->proj[L].data;
            if (q != QUANT_FP16) {
                qkvBufs[b++] = w->proj[L].scale;
                qkvBufs[b++] = w->proj[L].zero;
            }
            qkvBufs[b++] = w->theta;
            qkvBufs[b++] = st->qOut;
            qkvBufs[b++] = st->kCache[L];
            qkvBufs[b++] = st->vCache[L];
            if (q == QUANT_INT4) {
                qkvBufs[b++] = st->kScale[L];
                qkvBufs[b++] = st->kZero[L];
                qkvBufs[b++] = st->vScale[L];
                qkvBufs[b++] = st->vZero[L];
            }
            qkvBufs[b++] = st->position;
            int push[] = {1, MODEL_QKV_N, MODEL_K, MODEL_Q_OFF, MODEL_V_OFF};
            addOp(ops, &n, model_shader("RmsNorm-QKV", q), qkvBufs, b, push, 5, MODEL_QKV_N / 256, 1);

            buffer attBufs[9];
            int ba = 0;
            attBufs[ba++] = st->kCache[L];
            attBufs[ba++] = st->vCache[L];
            attBufs[ba++] = st->qOut;
            attBufs[ba++] = st->attnOut;
            if (q == QUANT_INT4) {
                attBufs[ba++] = st->kScale[L];
                attBufs[ba++] = st->kZero[L];
                attBufs[ba++] = st->vScale[L];
                attBufs[ba++] = st->vZero[L];
            }
            attBufs[ba++] = st->position;
            int push0[1] = {0};
            addOp(ops, &n, model_shader("Att-full", q), attBufs, ba, push0, 0, MODEL_HEADS, 1);

            addGemmAdd(r, ops, &n, &w->out[L], st->attnOut, st->h, st->h, q, 1);
        } else {
            if (L != 0) {
                addLinearProj(r, ops, &n, L, 0, 1);
            }

            buffer dnBufs[] = {st->qProj, st->kProj, st->vProj, st->gProj, st->aProj, st->bProj, st->stateS[L], st->yGated};
            int pushDn[] = {MODEL_N_V, MODEL_N_QK, MODEL_DIM};
            addOp(ops, &n, "GatedDeltaNet.spv", dnBufs, 8, pushDn, 3, MODEL_N_V, 1);

            addGemmAdd(r, ops, &n, &w->out[L], st->yGated, st->h, L == 0 ? st->embOut : st->h, q, 1);
        }

        buildFfn(r, ops, &n, L, 0, 1);
    }

    buildLmHead(r, ops, &n, &st->h, 1);

    execute(r->s, ops, n);

    uint32_t result = 0;
    readBuffer(r->s.dev.device, r->s.dev.physicalDevice, r->s.dev.queue, st->result, &result);
    return result;
}

void runGenerate(runner* r, const uint32_t* prompt, int nPrompt, int maxNewTokens) {
    uint32_t token = runPrefill(r, prompt, nPrompt);
    printf("gen[%d]= %u  pos= %u\n", 0, token, stateReadPosition(r->s, &r->st));
    for (int i = 1; i < maxNewTokens; i++) {
        token = runDecode(r, token);
        printf("gen[%d]= %u  pos= %u\n", i, token, stateReadPosition(r->s, &r->st));
    }
}

runner createRunner(session s, int maxM) {
    runner r = {0};
    r.s = s;
    r.maxM = maxM;
    r.w = createWeights(s);
    r.st = createState(s, maxM);
    return r;
}

void destroyRunner(runner* r) {
    destroyState(r->s, &r->st);
    destroyWeights(r->s, &r->w);
}