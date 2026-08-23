#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "generate.h"

static void addOp(operation* ops, int* n, const char* shader, int layer, buffer* bufs, int bc, const int* push, int pc, int dx, int dy) {
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
    op->layer = layer;
    (*n)++;
}

static void addGemvSplit(generator* g, operation* ops, int* n, int L, const tensor* wt, buffer input, buffer output, buffer residual, QuantType q) {
    buffer bufs[5];
    int b = 0;
    bufs[b++] = input;
    bufs[b++] = wt->data;
    bufs[b++] = g->st.gemvPartial;
    if (q != QUANT_FP16) {
        bufs[b++] = wt->scale;
        bufs[b++] = wt->zero;
    }
    int push[] = {1, MODEL_K, MODEL_K};
    addOp(ops, n, model_shader("GEMV-SplitK", q), L, bufs, b, push, 3, MODEL_K / 256, 4);

    buffer rBufs[3];
    rBufs[0] = g->st.gemvPartial;
    rBufs[1] = residual;
    rBufs[2] = output;
    int pushR[] = {MODEL_K, 4};
    addOp(ops, n, "Reduce-GEMV-ADD.spv", L, rBufs, 3, pushR, 2, MODEL_K / 256, 1);
}

static void addReduceGemvAdd(operation* ops, int* n, int L, buffer partial, buffer output, buffer residual) {
    buffer rBufs[3];
    rBufs[0] = partial;
    rBufs[1] = residual;
    rBufs[2] = output;
    int pushR[] = {MODEL_K, 4};
    addOp(ops, n, "Reduce-GEMV-ADD.spv", L, rBufs, 3, pushR, 2, MODEL_K / 256, 1);
}

static void addGemmAdd(generator* g, operation* ops, int* n, int L, const tensor* wt, buffer input, buffer output, buffer residual, QuantType q, int m) {
    if (m == 1) {
        addGemvSplit(g, ops, n, L, wt, input, output, residual, q);
        return;
    }

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
    int k = (input.buffer == g->st.act.buffer) ? MODEL_FFN_N : MODEL_K;
    int push[] = {m, MODEL_K, k};
    addOp(ops, n, model_shader("GEMM-ADD", q), L, bufs, b, push, 3, MODEL_K / 16, m / 16);
}

static void addLinearProj(generator* g, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    QuantType q = g->spec->layers[L].attn.q;

    if (!gemm) {
        buffer bufs[6];
        int b = 0;
        bufs[b++] = st->h;
        bufs[b++] = w->gammaIn[L];
        bufs[b++] = w->proj[L].data;
        if (q != QUANT_FP16) {
            bufs[b++] = w->proj[L].scale;
            bufs[b++] = w->proj[L].zero;
        }
        bufs[b++] = st->linprojPartial;
        int push[] = {m, MODEL_PROJ_N, MODEL_K};
        addOp(ops, n, model_shader("RmsNorm-LinearProj-SplitK", q), L, bufs, b, push, 3,
              (MODEL_PROJ_N + 255) / 256, 4);

        buffer rBufs[7];
        rBufs[0] = st->linprojPartial;
        rBufs[1] = st->qProj;
        rBufs[2] = st->kProj;
        rBufs[3] = st->vProj;
        rBufs[4] = st->gProj;
        rBufs[5] = st->aProj;
        rBufs[6] = st->bProj;
        int pushR[] = {MODEL_PROJ_N};
        addOp(ops, n, "Reduce-LinearProj.spv", L, rBufs, 7, pushR, 1,
              (MODEL_PROJ_N + 255) / 256, 1);
        return;
    }

    buffer bufs[14];
    int b = 0;
    bufs[b++] = st->h;
    bufs[b++] = w->gammaIn[L];
    bufs[b++] = w->proj[L].data;
    bufs[b++] = st->qProj;
    bufs[b++] = st->kProj;
    bufs[b++] = st->vProj;
    bufs[b++] = st->gProj;
    bufs[b++] = st->aProj;
    bufs[b++] = st->bProj;
    if (q != QUANT_FP16) {
        bufs[b++] = w->proj[L].scale;
        bufs[b++] = w->proj[L].zero;
    }
    int push[] = {m, MODEL_PROJ_N, MODEL_K};
    addOp(ops, n, model_shader("RmsNorm-LinearProj-GEMM", q), L, bufs, b, push, 3,
          MODEL_PROJ_N / 16, m / 16);
}

static void buildFfn(generator* g, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    QuantType f = g->spec->layers[L].ffn.q;

    if (!gemm) {
        buffer upBufs[9];
        int b = 0;
        upBufs[b++] = st->h;
        upBufs[b++] = w->gammaF[L];
        upBufs[b++] = w->gate[L].data;
        upBufs[b++] = w->up[L].data;
        upBufs[b++] = st->ffnPartial;
        if (f != QUANT_FP16) {
            upBufs[b++] = w->gate[L].scale;
            upBufs[b++] = w->gate[L].zero;
            upBufs[b++] = w->up[L].scale;
            upBufs[b++] = w->up[L].zero;
        }
        int push[] = {m, MODEL_FFN_N, MODEL_K};
        addOp(ops, n, model_shader("RmsNorm-up-ffn-SplitK", f), L, upBufs, b, push, 3,
              MODEL_FFN_N / 256, 4);

        buffer dBufs[5];
        int d = 0;
        dBufs[d++] = st->ffnPartial;
        dBufs[d++] = w->down[L].data;
        dBufs[d++] = st->gemvPartial;
        if (f != QUANT_FP16) {
            dBufs[d++] = w->down[L].scale;
            dBufs[d++] = w->down[L].zero;
        }
        int pushD[] = {m, MODEL_K, MODEL_FFN_N};
        addOp(ops, n, model_shader("FFN-Down-SplitK", f), L, dBufs, d, pushD, 3,
              MODEL_K / 256, 4);

        addReduceGemvAdd(ops, n, L, st->gemvPartial, st->h, st->h);
        return;
    }

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
    addOp(ops, n, model_shader("RmsNorm-swiglu-ffn-GEMM", f), L, ffnBufs, bf, push, 3,
          MODEL_FFN_N / 16, m / 16);

    addGemmAdd(g, ops, n, L, &w->down[L], st->act, st->h, st->h, f, m);
}

static void buildAttention(generator* g, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    QuantType q = g->spec->layers[L].attn.q;

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
    if (q != QUANT_FP16) {
        qkvBufs[b++] = st->kScale[L];
        qkvBufs[b++] = st->kZero[L];
        qkvBufs[b++] = st->vScale[L];
        qkvBufs[b++] = st->vZero[L];
    }
    qkvBufs[b++] = st->position;

    if (gemm) {
        int push[] = {m, MODEL_QKV_N, MODEL_K, 0, MODEL_Q_OFF, MODEL_V_OFF};
        addOp(ops, n, model_shader("RmsNorm-QKV-GEMM", q), L, qkvBufs, b, push, 6, MODEL_HEADS + 2 * MODEL_KV_HEADS, m / 16);

        buffer attBufs[9];
        int ba = 0;
        attBufs[ba++] = st->kCache[L];
        attBufs[ba++] = st->vCache[L];
        attBufs[ba++] = st->qOut;
        attBufs[ba++] = st->attnOut;
        if (q != QUANT_FP16) {
            attBufs[ba++] = st->kScale[L];
            attBufs[ba++] = st->kZero[L];
            attBufs[ba++] = st->vScale[L];
            attBufs[ba++] = st->vZero[L];
        }
        int pushA[] = {m};
        addOp(ops, n, model_shader("Att-full-GEMM", q), L, attBufs, ba, pushA, 1, MODEL_HEADS, m / 16);
    } else {
        buffer splitBufs[6];
        int bs = 0;
        splitBufs[bs++] = st->h;
        splitBufs[bs++] = w->gammaIn[L];
        splitBufs[bs++] = w->proj[L].data;
        if (q != QUANT_FP16) {
            splitBufs[bs++] = w->proj[L].scale;
            splitBufs[bs++] = w->proj[L].zero;
        }
        splitBufs[bs++] = st->qkvPartial;
        int push[] = {1, MODEL_QKV_N, MODEL_K};
        addOp(ops, n, model_shader("RmsNorm-QKV-SplitK", q), L, splitBufs, bs, push, 3,
              MODEL_QKV_N / 256, 4);

        buffer ropeBufs[10];
        int br = 0;
        ropeBufs[br++] = st->qkvPartial;
        ropeBufs[br++] = st->qOut;
        ropeBufs[br++] = st->kCache[L];
        ropeBufs[br++] = st->vCache[L];
        if (q != QUANT_FP16) {
            ropeBufs[br++] = st->kScale[L];
            ropeBufs[br++] = st->kZero[L];
            ropeBufs[br++] = st->vScale[L];
            ropeBufs[br++] = st->vZero[L];
        }
        ropeBufs[br++] = w->theta;
        ropeBufs[br++] = st->position;
        int pushRope[] = {MODEL_QKV_N, MODEL_Q_OFF, MODEL_V_OFF};
        addOp(ops, n, model_shader("Reduce-Rope", q), L, ropeBufs, br, pushRope, 3,
              MODEL_HEADS + 2 * MODEL_KV_HEADS, 1);

        buffer attBufs[9];
        int ba = 0;
        attBufs[ba++] = st->kCache[L];
        attBufs[ba++] = st->vCache[L];
        attBufs[ba++] = st->qOut;
        attBufs[ba++] = st->attnOut;
        if (q != QUANT_FP16) {
            attBufs[ba++] = st->kScale[L];
            attBufs[ba++] = st->kZero[L];
            attBufs[ba++] = st->vScale[L];
            attBufs[ba++] = st->vZero[L];
        }
        attBufs[ba++] = st->position;
        int push0[1] = {0};
        addOp(ops, n, model_shader("Att-full", q), L, attBufs, ba, push0, 0, MODEL_HEADS, 1);
    }

    addGemmAdd(g, ops, n, L, &w->out[L], st->attnOut, st->h, st->h, q, m);
}

static void buildDelta(generator* g, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    QuantType q = g->spec->layers[L].attn.q;

    if (L != 0) {
        addLinearProj(g, ops, n, L, gemm, m);
    }

    buffer dnBufs[] = {st->qProj, st->kProj, st->vProj, st->gProj, st->aProj, st->bProj, st->stateS[L], st->yGated};
    if (gemm) {
        int pushDn[] = {m};
        addOp(ops, n, "GatedDeltaNet-GEMM.spv", L, dnBufs, 8, pushDn, 1, MODEL_N_V, 1);
    } else {
        int pushDn[] = {MODEL_N_V, MODEL_N_QK, MODEL_DIM};
        addOp(ops, n, "GatedDeltaNet.spv", L, dnBufs, 8, pushDn, 3, MODEL_N_V, 1);
    }

    addGemmAdd(g, ops, n, L, &w->out[L], st->yGated, st->h, L == 0 ? st->embOut : st->h, q, m);
}

static void buildLayer(generator* g, operation* ops, int* n, int L, int gemm, int m) {
    const layer* ly = &g->spec->layers[L];
    if (ly->attn.type == ATTENTION_FULL) {
        buildAttention(g, ops, n, L, gemm, m);
    } else if (ly->attn.type == ATTENTION_DELTA) {
        buildDelta(g, ops, n, L, gemm, m);
    }
    if (ly->ffn.type == FFN_SWIGLU) {
        buildFfn(g, ops, n, L, gemm, m);
    }
}

static void buildLmHead(generator* g, operation* ops, int* n, buffer* input, int doIncrement) {
    model_state* st = &g->st;
    model_weights* w = &g->w;

    buffer lmBufs[] = {*input, w->lmHead, st->maxValue, st->maxIndex, w->gammaFinal};
    int pushL[] = {1, MODEL_VOCAB, MODEL_K};
    addOp(ops, n, "LMHead-GEMV-ArgMax-FP16.spv", -1, lmBufs, 5, pushL, 3, (MODEL_VOCAB + 255) / 256, 1);

    buffer redBufs[] = {st->maxValue, st->maxIndex, st->result, st->position};
    int pushR[] = {MODEL_VOCAB, doIncrement};
    addOp(ops, n, "ArgMax-Reduce.spv", -1, redBufs, 4, pushR, 2, 1, 1);
}

static int compileDecode(generator* g) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    operation* ops = g->decodeOps;
    int n = 0;

    buffer embedBufs[] = {st->tokenIds, w->embed, w->gammaIn[0], w->proj[0].data, st->qProj, st->kProj, st->vProj, st->gProj, st->aProj, st->bProj, st->embOut};
    int pushE[] = {1, MODEL_PROJ_N, MODEL_K, MODEL_VOCAB};
    addOp(ops, &n, "Embed-RmsNorm-LinearProj-FP16.spv", -1, embedBufs, 11, pushE, 4, (MODEL_PROJ_N + 255) / 256, 1);

    for (int L = 0; L < g->spec->layerCount; L++) {
        buildLayer(g, ops, &n, L, 0, 1);
    }

    buildLmHead(g, ops, &n, &st->h, 1);
    return n;
}

static int compilePrefill(generator* g) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    operation* ops = g->prefillOps;
    int n = 0;
    int m = g->maxM;

    buffer embedBufs[] = {st->tokenIds, w->embed, w->gammaIn[0], w->proj[0].data, st->qProj, st->kProj, st->vProj, st->gProj, st->aProj, st->bProj, st->embOut};
    int pushE[] = {m, MODEL_PROJ_N, MODEL_K, MODEL_VOCAB};
    addOp(ops, &n, "Embed-RmsNorm-LinearProj-GEMM-FP16.spv", -1, embedBufs, 11, pushE, 4, MODEL_PROJ_N / 16, m / 16);

    for (int L = 0; L < g->spec->layerCount; L++) {
        buildLayer(g, ops, &n, L, 1, m);
    }

    return n;
}

static int compileFinal(generator* g) {
    int n = 0;
    buildLmHead(g, g->finalOps, &n, &g->st.lastRow, 0);
    return n;
}

generator* createGenerator(session s, const model_config* spec, int maxM) {
    static generator g;
    memset(&g, 0, sizeof(generator));
    g.s = s;
    g.spec = spec;
    g.maxM = maxM;
    g.w = createWeights(s, spec);
    g.st = createState(s, spec, maxM);
    g.decodeOpCount = compileDecode(&g);
    g.prefillOpCount = compilePrefill(&g);
    g.finalOpCount = compileFinal(&g);
    return &g;
}

void destroyGenerator(generator* g) {
    destroyState(g->s, &g->st);
    destroyWeights(g->s, &g->w);
}

uint32_t runPrefill(generator* g, const uint32_t* tokens, int nTokens) {
    model_state* st = &g->st;
    int m = g->maxM;

    memcpy(st->tokenIds.mappedMemory, tokens, sizeof(uint32_t) * m);

    executeLogged(g->s, g->prefillOps, g->prefillOpCount, "prefill", (int)g->nextPos);

    float* hCpu = (float*)malloc(sizeof(float) * m * MODEL_K);
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->h, hCpu);
    memcpy(st->lastRow.mappedMemory, hCpu + (nTokens - 1) * MODEL_K, sizeof(float) * MODEL_K);
    free(hCpu);

    executeLogged(g->s, g->finalOps, g->finalOpCount, "prefill", (int)g->nextPos);

    g->nextPos += (uint32_t)m;

    uint32_t result = 0;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->result, &result);
    return result;
}

uint32_t runDecode(generator* g, uint32_t token) {
    model_state* st = &g->st;

    ((uint32_t*)st->tokenIds.mappedMemory)[0] = token;

    executeLogged(g->s, g->decodeOps, g->decodeOpCount, "decode", (int)g->nextPos);

    g->nextPos++;

    uint32_t result = 0;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->result, &result);
    return result;
}

void runGenerate(generator* g, const uint32_t* prompt, int nPrompt, int maxNewTokens) {
    uint64_t tPrefill = GetTickCount64();
    uint32_t token = runPrefill(g, prompt, nPrompt);
    uint64_t prefillMs = GetTickCount64() - tPrefill;
    double prefillSpeed = prefillMs > 0 ? (double)(nPrompt * 1000) / prefillMs : 0.0;
    printf("gen[%d]: %u | pos: %u | speed: %.2f token/s\n", 0, token, stateReadPosition(g->s, &g->st), prefillSpeed);
    uint64_t t0 = GetTickCount64();
    for (int i = 1; i < maxNewTokens; i++) {
        token = runDecode(g, token);
        uint64_t ms = GetTickCount64() - t0;
        double speed = ms > 0 ? (double)(i * 1000) / ms : 0.0;
        printf("gen[%d]: %u | pos: %u | speed: %.2f token/s\n", i, token, stateReadPosition(g->s, &g->st), speed);
    }
    closeTimingLog();
}
