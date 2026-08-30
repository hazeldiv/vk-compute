#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "generate.h"

#define ATT_SPLIT_THRESHOLD 256
#define FIRST_TOKEN_DELAY_MS 25.0

static double g_qpcFreqMs = 0.0;

static double nowMs(void) {
    if (g_qpcFreqMs == 0.0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        g_qpcFreqMs = (double)f.QuadPart / 1000.0;
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / g_qpcFreqMs;
}

static void sleepMs(double ms) {
    if (ms <= 0.0) return;
    double target = nowMs() + ms;
    double coarse = ms - 1.0;
    if (coarse > 0.0) Sleep((DWORD)coarse);
    while (nowMs() < target) {
    }
}

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
    int tn = (q == QUANT_INT4) ? 64 : 32;
    int push[] = {m, MODEL_K, k};
    addOp(ops, n, model_shader("GEMM-ADD2", q), L, bufs, b, push, 3, MODEL_K / tn, (m + 15) / 16);
}

static void addLinearProj(generator* g, operation* ops, int* n, int L, int gemm, int m, buffer input) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    QuantType q = g->spec->layers[L].attn.q;

    if (!gemm) {
        buffer bufs[6];
        int b = 0;
        bufs[b++] = input;
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
        rBufs[4] = st->zProj;
        rBufs[5] = st->aProj;
        rBufs[6] = st->bProj;
        int pushR[] = {MODEL_PROJ_N};
        addOp(ops, n, "Reduce-LinearProj.spv", L, rBufs, 7, pushR, 1,
              (MODEL_PROJ_N + 255) / 256, 1);
        return;
    }

    buffer bufs[14];
    int b = 0;
bufs[b++] = input;
        bufs[b++] = w->gammaIn[L];
        bufs[b++] = w->proj[L].data;
        bufs[b++] = st->qProj;
        bufs[b++] = st->kProj;
        bufs[b++] = st->vProj;
        bufs[b++] = st->zProj;
        bufs[b++] = st->aProj;
        bufs[b++] = st->bProj;
        if (q != QUANT_FP16) {
            bufs[b++] = w->proj[L].scale;
            bufs[b++] = w->proj[L].zero;
        }
        bufs[b++] = st->invRms;
    int push[] = {m, MODEL_PROJ_N, MODEL_K};
    int pushP[] = {MODEL_K};
    buffer proBufs[2];
    proBufs[0] = input;
    proBufs[1] = st->invRms;
    addOp(ops, n, "RmsNorm-Prologue.spv", L, proBufs, 2, pushP, 1, m, 1);
    addOp(ops, n, model_shader("RmsNorm-LinearProj-GEMM2", q), L, bufs, b, push, 3,
          MODEL_PROJ_N / 32, (m + 15) / 16);
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
        int push[] = {m, MODEL_FFN_N, MODEL_K, MODEL_FFN_N};
        addOp(ops, n, model_shader("RmsNorm-up-ffn-SplitK", f), L, upBufs, b, push, 4,
              2 * MODEL_FFN_N / 256, 4);

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

    int pushP[] = {MODEL_K};
    buffer proBufs[2];
    proBufs[0] = st->h;
    proBufs[1] = st->invRms;
    addOp(ops, n, "RmsNorm-Prologue.spv", L, proBufs, 2, pushP, 1, m, 1);

    if (f == QUANT_FP16) {
        buffer ffnBufs[6];
        int bf = 0;
        ffnBufs[bf++] = st->h;
        ffnBufs[bf++] = w->gammaF[L];
        ffnBufs[bf++] = w->gate[L].data;
        ffnBufs[bf++] = w->up[L].data;
        ffnBufs[bf++] = st->act;
        ffnBufs[bf++] = st->invRms;
        int push[] = {m, MODEL_FFN_N, MODEL_K};
        addOp(ops, n, model_shader("RmsNorm-swiglu-ffn-GEMM2", f), L, ffnBufs, bf, push, 3,
              MODEL_FFN_N / 32, (m + 15) / 16);
    } else {
        buffer flatBufs[11];
        int bf2 = 0;
        flatBufs[bf2++] = st->h;
        flatBufs[bf2++] = w->gammaF[L];
        flatBufs[bf2++] = w->gate[L].data;
        flatBufs[bf2++] = w->up[L].data;
        flatBufs[bf2++] = st->gAct;
        flatBufs[bf2++] = st->uAct;
        flatBufs[bf2++] = w->gate[L].scale;
        flatBufs[bf2++] = w->gate[L].zero;
        flatBufs[bf2++] = w->up[L].scale;
        flatBufs[bf2++] = w->up[L].zero;
        flatBufs[bf2++] = st->invRms;
        int pushF[] = {m, MODEL_FFN_N, MODEL_K, MODEL_FFN_N};
        addOp(ops, n, model_shader("RmsNorm-swiglu-flat-GEMM2", f), L, flatBufs, bf2, pushF, 4,
              (MODEL_FFN_N * 2) / 32, (m + 15) / 16);

        buffer cmbBufs[3];
        cmbBufs[0] = st->gAct;
        cmbBufs[1] = st->uAct;
        cmbBufs[2] = st->act;
        int pushC[] = {m * MODEL_FFN_N};
        addOp(ops, n, "Swiglu-combine.spv", L, cmbBufs, 3, pushC, 1, (m * MODEL_FFN_N + 255) / 256, 1);
    }

    addGemmAdd(g, ops, n, L, &w->down[L], st->act, st->h, st->h, f, m);
}

static void buildAttention(generator* g, operation* ops, int* n, int L, int gemm, int m, int splitAttn, int ctx, int offset) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    QuantType q = g->spec->layers[L].attn.q;

    if (gemm) {
        int pushP[] = {MODEL_K};
        buffer proBufs[2];
        proBufs[0] = st->h;
        proBufs[1] = st->invRms;
        addOp(ops, n, "RmsNorm-Prologue.spv", L, proBufs, 2, pushP, 1, m, 1);

        buffer qkvGBufs[7];
        int bq2 = 0;
        qkvGBufs[bq2++] = st->h;
        qkvGBufs[bq2++] = w->gammaIn[L];
        qkvGBufs[bq2++] = w->proj[L].data;
        if (q != QUANT_FP16) {
            qkvGBufs[bq2++] = w->proj[L].scale;
            qkvGBufs[bq2++] = w->proj[L].zero;
        }
        qkvGBufs[bq2++] = st->qkvRaw;
        qkvGBufs[bq2++] = st->invRms;
        int tnQkv = (q == QUANT_INT4) ? 64 : 32;
        int pushQ[] = {m, MODEL_QKV_N, MODEL_K};
        addOp(ops, n, model_shader("RmsNorm-QKV-GEMM2", q), L, qkvGBufs, bq2, pushQ, 3,
              MODEL_QKV_N / tnQkv, (m + 15) / 16);

        buffer ropeGBufs[13];
        int brg = 0;
        ropeGBufs[brg++] = st->qkvRaw;
        ropeGBufs[brg++] = st->qOut;
        ropeGBufs[brg++] = st->kCache[L];
        ropeGBufs[brg++] = st->vCache[L];
        if (q != QUANT_FP16) {
            ropeGBufs[brg++] = st->kScale[L];
            ropeGBufs[brg++] = st->kZero[L];
            ropeGBufs[brg++] = st->vScale[L];
            ropeGBufs[brg++] = st->vZero[L];
        }
        ropeGBufs[brg++] = w->theta;
        ropeGBufs[brg++] = w->qNorm[L];
        ropeGBufs[brg++] = w->kNorm[L];
        ropeGBufs[brg++] = st->gAttn;
        int pushRG[] = {MODEL_QKV_N, MODEL_G_OFF, MODEL_K_OFF, MODEL_V_OFF, offset};
        addOp(ops, n, model_shader("Rope-GEMM", q), L, ropeGBufs, brg, pushRG, 5,
              2 * MODEL_HEADS + 2 * MODEL_KV_HEADS, m);

        for (int hb = 0; hb < MODEL_HEADS / MODEL_KV_HEADS; hb++) {
            int pushA[] = {ctx, offset, m, hb * MODEL_KV_HEADS};

            buffer qkBufs[5];
            int bq = 0;
            qkBufs[bq++] = st->qOut;
            qkBufs[bq++] = st->kCache[L];
            if (q != QUANT_FP16) {
                qkBufs[bq++] = st->kScale[L];
                qkBufs[bq++] = st->kZero[L];
            }
            qkBufs[bq++] = st->attScores;
            addOp(ops, n, model_shader("Att-QK2", q), L, qkBufs, bq, pushA, 4, (ctx + 63) / 64, ((m + 15) / 16) * 4);

            buffer smBufs[2];
            smBufs[0] = st->attScores;
            smBufs[1] = st->smSum;
            addOp(ops, n, "Att-Softmax.spv", L, smBufs, 2, pushA, 4, m, 4);

            buffer pvBufs[6];
            int bp = 0;
            pvBufs[bp++] = st->attScores;
            pvBufs[bp++] = st->vCache[L];
            if (q != QUANT_FP16) {
                pvBufs[bp++] = st->vScale[L];
                pvBufs[bp++] = st->vZero[L];
            }
            pvBufs[bp++] = st->attnOut;
            pvBufs[bp++] = st->smSum;
            addOp(ops, n, model_shader("Att-PV2", q), L, pvBufs, bp, pushA, 4, 4, ((m + 15) / 16) * 4);
        }

        buffer gateBufs[2];
        gateBufs[0] = st->gAttn;
        gateBufs[1] = st->attnOut;
        int pushGate[] = {m * MODEL_Q_OFF};
        addOp(ops, n, "Gate-Sigmoid.spv", L, gateBufs, 2, pushGate, 1,
              (m * MODEL_Q_OFF + 255) / 256, 1);
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

        buffer ropeBufs[13];
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
        ropeBufs[br++] = w->qNorm[L];
        ropeBufs[br++] = w->kNorm[L];
        ropeBufs[br++] = st->gAttn;
        int pushRope[] = {MODEL_QKV_N, MODEL_G_OFF, MODEL_K_OFF, MODEL_V_OFF};
        addOp(ops, n, model_shader("Reduce-Rope", q), L, ropeBufs, br, pushRope, 4,
              2 * MODEL_HEADS + 2 * MODEL_KV_HEADS, 1);

        buffer attBufs[10];
        int ba = 0;
        attBufs[ba++] = st->kCache[L];
        attBufs[ba++] = st->vCache[L];
        attBufs[ba++] = st->qOut;
        if (q != QUANT_FP16) {
            attBufs[ba++] = st->kScale[L];
            attBufs[ba++] = st->kZero[L];
            attBufs[ba++] = st->vScale[L];
            attBufs[ba++] = st->vZero[L];
        }
        attBufs[ba++] = st->attPartial;
        attBufs[ba++] = st->position;
        int push0[1] = {0};
        if (splitAttn) {
            addOp(ops, n, model_shader("Att-SplitK2", q), L, attBufs, ba, push0, 0, MODEL_HEADS, 128);

            buffer attRedBufs[3];
            attRedBufs[0] = st->attPartial;
            attRedBufs[1] = st->attnOut;
            attRedBufs[2] = st->position;
            addOp(ops, n, "Reduce-Att2.spv", L, attRedBufs, 3, push0, 0, MODEL_HEADS * MODEL_HEAD_DIM / 256, 1);
        } else {
            buffer fullBufs[10];
            int bf = 0;
            fullBufs[bf++] = st->kCache[L];
            fullBufs[bf++] = st->vCache[L];
            fullBufs[bf++] = st->qOut;
            fullBufs[bf++] = st->attnOut;
            if (q != QUANT_FP16) {
                fullBufs[bf++] = st->kScale[L];
                fullBufs[bf++] = st->kZero[L];
                fullBufs[bf++] = st->vScale[L];
                fullBufs[bf++] = st->vZero[L];
            }
            fullBufs[bf++] = st->position;
            addOp(ops, n, model_shader("Att-full", q), L, fullBufs, bf, push0, 0, MODEL_HEADS, 1);
        }

        buffer gateBufs[2];
        gateBufs[0] = st->gAttn;
        gateBufs[1] = st->attnOut;
        int pushGate[] = {MODEL_Q_OFF};
        addOp(ops, n, "Gate-Sigmoid.spv", L, gateBufs, 2, pushGate, 1,
              (MODEL_Q_OFF + 255) / 256, 1);
    }

    addGemmAdd(g, ops, n, L, &w->out[L], st->attnOut, st->h, st->h, q, m);
}

static void buildDelta(generator* g, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    QuantType q = g->spec->layers[L].attn.q;

    if (L != 0) {
        addLinearProj(g, ops, n, L, gemm, m, st->h);
    }

    buffer convBufs[] = {st->qProj, st->kProj, st->vProj, w->conv[L], st->convHist[L]};
    int pushConv[] = {m};
    addOp(ops, n, "Conv-SiLU.spv", L, convBufs, 5, pushConv, 1, MODEL_ZQKV_N / 256, 1);

    buffer dnBufs[] = {st->qProj, st->kProj, st->vProj, st->aProj, st->bProj, st->stateS[L], st->yGated, st->zProj, w->attnNorm[L], w->aLog[L], w->dtBias[L]};
    if (gemm) {
        int pushDn[] = {m};
        addOp(ops, n, "GatedDeltaNet-GEMM.spv", L, dnBufs, 11, pushDn, 1, MODEL_N_V, 1);
    } else {
        int pushDn[] = {MODEL_N_V, MODEL_N_QK, MODEL_DIM};
        addOp(ops, n, "GatedDeltaNet.spv", L, dnBufs, 11, pushDn, 3, MODEL_N_V, 1);
    }

    addGemmAdd(g, ops, n, L, &w->out[L], st->yGated, st->h, L == 0 ? st->embOut : st->h, q, m);
}

static void buildLayer(generator* g, operation* ops, int* n, int L, int gemm, int m, int splitAttn, int ctx, int offset) {
    const layer* ly = &g->spec->layers[L];
    if (ly->attn.type == ATTENTION_FULL) {
        buildAttention(g, ops, n, L, gemm, m, splitAttn, ctx, offset);
    } else if (ly->attn.type == ATTENTION_DELTA) {
        buildDelta(g, ops, n, L, gemm, m);
    }
    if (ly->ffn.type == FFN_SWIGLU) {
        buildFfn(g, ops, n, L, gemm, m);
    }
}

static void buildLmHead(generator* g, operation* ops, int* n, buffer* input, int doIncrement, int passIdx) {
    model_state* st = &g->st;
    model_weights* w = &g->w;

    if (g->sampling) {
        buffer lmBufs[] = {*input, w->lmHead, st->logits, w->gammaFinal};
        int pushL[] = {1, g->vocab, MODEL_K};
        addOp(ops, n, "LMHead-GEMV-FP16.spv", -1, lmBufs, 4, pushL, 3, (g->vocab + 255) / 256, 1);
    } else {
        buffer lmBufs[] = {*input, w->lmHead, st->maxValue, st->maxIndex, w->gammaFinal};
        int pushL[] = {1, g->vocab, MODEL_K};
        addOp(ops, n, "LMHead-GEMV-ArgMax-FP16.spv", -1, lmBufs, 5, pushL, 3, (g->vocab + 255) / 256, 1);
    }

    buffer redBufs[] = {st->maxValue, st->maxIndex, st->logits, st->sampleParams, st->sampleHistory, st->sampleRng, st->result, st->position, st->tokenIds};
    int pushR[] = {g->vocab, doIncrement, passIdx, g->sampling};
    addOp(ops, n, "ArgMax-Reduce.spv", -1, redBufs, 9, pushR, 4, 1, 1);
}

static int compileDecodeGroup(generator* g, operation* ops, int splitAttn) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    int n = 0;

    for (int p = 0; p < DECODE_GROUP; p++) {
        buffer embedBufs[] = {st->tokenIds, w->embed, w->gammaIn[0], w->proj[0].data, st->qProj, st->kProj, st->vProj, st->zProj, st->aProj, st->bProj, st->embOut};
        int pushE[] = {1, MODEL_PROJ_N, MODEL_K, g->vocab};
        addOp(ops, &n, "Embed-RmsNorm-LinearProj-FP16.spv", -1, embedBufs, 11, pushE, 4, (MODEL_PROJ_N + 255) / 256, 1);

        for (int L = 0; L < g->spec->layerCount; L++) {
            buildLayer(g, ops, &n, L, 0, 1, splitAttn, 0, 0);
        }

        buildLmHead(g, ops, &n, &st->h, 1, p);
    }
    return n;
}

static int compilePrefillHead(generator* g, operation* ops, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    int n = 0;

    buffer gatherBufs[4];
    gatherBufs[0] = st->tokenIds;
    gatherBufs[1] = w->embed;
    gatherBufs[2] = st->embStaged;
    gatherBufs[3] = st->embOut;
    int pushG[] = {g->vocab};
    addOp(ops, &n, "Embed-Gather.spv", -1, gatherBufs, 4, pushG, 1, m, 1);

    addLinearProj(g, ops, &n, 0, 1, m, st->embStaged);

    return n;
}

static int compilePrefill(generator* g, int m, int offset) {
    operation* ops = g->prefillOps;
    int n = compilePrefillHead(g, ops, m);
    for (int L = 0; L < g->spec->layerCount; L++) {
        buildLayer(g, ops, &n, L, 1, m, 0, offset + m, offset);
    }
    return n;
}

static int compileFinal(generator* g) {
    int n = 0;
    buildLmHead(g, g->finalOps, &n, &g->st.lastRow, 0, 0);
    return n;
}

generator* createGenerator(session s, const model_config* spec, int maxM, const char* weightDir, const char* customHead, const char* customEmbed, int eos, int maxCtx, int verboseWeights) {
    static generator g;
    memset(&g, 0, sizeof(generator));
    g.s = s;
    g.spec = spec;
    g.maxM = maxM;
    g.eos = eos;
    g.maxCtx = maxCtx;
    g.w = createWeights(s, spec, weightDir, customHead, customEmbed, verboseWeights);
    g.vocab = g.w.vocab;
    g.st = createState(s, spec, maxM, g.vocab, verboseWeights);
    g.groupOpCount = compileDecodeGroup(&g, g.groupOps, 1);
    g.groupOpCountShort = compileDecodeGroup(&g, g.groupOpsShort, 0);
    g.prefillOpCount = compilePrefill(&g, g.maxM, 0);
    g.finalOpCount = compileFinal(&g);
    return &g;
}

void destroyGenerator(generator* g) {
    destroyState(g->s, &g->st);
    destroyWeights(g->s, &g->w);
}

static void dumpBuffer(generator* g, buffer buf, const char* name, int64_t floats) {
    if (g->dumpDir[0] == '\0') return;
    char path[320];
    snprintf(path, sizeof(path), "%s/%s.bin", g->dumpDir, name);
    float* host = (float*)malloc((size_t)buf.size);
    if (host == NULL) return;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, buf, host);
    FILE* f = fopen(path, "wb");
    if (f != NULL) {
        int64_t avail = (int64_t)buf.size / (int64_t)sizeof(float);
        int64_t n = floats < avail ? floats : avail;
        fwrite(host, sizeof(float), (size_t)n, f);
        fclose(f);
    }
    free(host);
}

static void dumpBufferFp16(generator* g, buffer buf, const char* name, int64_t halves) {
    if (g->dumpDir[0] == '\0') return;
    char path[320];
    snprintf(path, sizeof(path), "%s/%s.bin", g->dumpDir, name);
    uint16_t* host = (uint16_t*)malloc((size_t)buf.size);
    if (host == NULL) return;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, buf, host);
    FILE* f = fopen(path, "wb");
    if (f != NULL) {
        int64_t avail = (int64_t)buf.size / (int64_t)sizeof(uint16_t);
        int64_t n = halves < avail ? halves : avail;
        fwrite(host, sizeof(uint16_t), (size_t)n, f);
        fclose(f);
    }
    free(host);
}

static void dumpBufferU8(generator* g, buffer buf, const char* name, int64_t bytes) {
    if (g->dumpDir[0] == '\0') return;
    char path[320];
    snprintf(path, sizeof(path), "%s/%s.bin", g->dumpDir, name);
    uint8_t* host = (uint8_t*)malloc((size_t)buf.size);
    if (host == NULL) return;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, buf, host);
    FILE* f = fopen(path, "wb");
    if (f != NULL) {
        int64_t avail = (int64_t)buf.size;
        int64_t n = bytes < avail ? bytes : avail;
        fwrite(host, 1, (size_t)n, f);
        fclose(f);
    }
    free(host);
}

static void executeChunked(session s, operation* ops, int opCount, const char* phase, int token) {
    int done = 0;
    while (done < opCount) {
        int cnt = opCount - done;
        if (cnt > 8) cnt = 8;
        executeRecord(&s, ops + done, cnt);
        executeSubmitNow(&s);
        executeWaitLast(&s);
        logLastFrame(&s, ops + done, cnt, phase, token);
        done += cnt;
    }
}

uint32_t runPrefill(generator* g, const uint32_t* tokens, int nTokens) {
    model_state* st = &g->st;
    int m = g->maxM;

    int done = 0;
    int lastCur = 0;
    while (done < nTokens) {
        int cur = nTokens - done;
        if (cur > m) cur = m;
        int padded = (cur + 15) & ~15;
        memcpy(st->tokenIds.mappedMemory, tokens + done, sizeof(uint32_t) * cur);
        for (int i = cur; i < padded; i++) ((uint32_t*)st->tokenIds.mappedMemory)[i] = 0;

        if (g->dumpLayers > 0) {
            int headCount = compilePrefillHead(g, g->prefillOps, padded);
            executeChunked(g->s, g->prefillOps, headCount, "prefill", (int)(g->nextPos + done));
            dumpBuffer(g, st->embOut, "layer_00_embed", (int64_t)padded * MODEL_K);
            for (int L = 0; L < g->spec->layerCount && L < g->dumpLayers; L++) {
                const layer* ly = &g->spec->layers[L];
                int n = 0;
                if (ly->attn.type == ATTENTION_FULL) {
                    buildAttention(g, g->prefillOps, &n, L, 1, padded, 0, done + padded, done);
                } else if (ly->attn.type == ATTENTION_DELTA) {
                    buildDelta(g, g->prefillOps, &n, L, 1, padded);
                }
                executeChunked(g->s, g->prefillOps, n, "prefill", (int)(g->nextPos + done));
                char name[64];
                snprintf(name, sizeof(name), "layer_%02d_hattn", L + 1);
                dumpBuffer(g, st->h, name, (int64_t)padded * MODEL_K);

                int nf = 0;
                if (ly->ffn.type == FFN_SWIGLU) {
                    buildFfn(g, g->prefillOps, &nf, L, 1, padded);
                }
                executeChunked(g->s, g->prefillOps, nf, "prefill", (int)(g->nextPos + done));
                snprintf(name, sizeof(name), "layer_%02d_h", L + 1);
                dumpBuffer(g, st->h, name, (int64_t)padded * MODEL_K);
                if (g->spec->layers[L].attn.type == ATTENTION_FULL) {
                    snprintf(name, sizeof(name), "layer_%02d_qkvRaw", L + 1);
                    dumpBuffer(g, st->qkvRaw, name, (int64_t)padded * MODEL_QKV_N);
                    snprintf(name, sizeof(name), "layer_%02d_qOut", L + 1);
                    dumpBuffer(g, st->qOut, name, (int64_t)padded * MODEL_Q_OFF);
                    snprintf(name, sizeof(name), "layer_%02d_attnOut", L + 1);
                    dumpBuffer(g, st->attnOut, name, (int64_t)padded * MODEL_Q_OFF);
                    snprintf(name, sizeof(name), "layer_%02d_gAttn", L + 1);
                    dumpBuffer(g, st->gAttn, name, (int64_t)padded * MODEL_Q_OFF);
                    snprintf(name, sizeof(name), "layer_%02d_scores", L + 1);
                    dumpBufferFp16(g, st->attScores, name, (int64_t)4 * padded * MODEL_MAX_CTX);
                    snprintf(name, sizeof(name), "layer_%02d_smSum", L + 1);
                    dumpBuffer(g, st->smSum, name, (int64_t)4 * padded);
                    QuantType aq = g->spec->layers[L].attn.q;
                    if (aq == QUANT_FP16) {
                        snprintf(name, sizeof(name), "layer_%02d_kCache", L + 1);
                        dumpBufferFp16(g, st->kCache[L], name, (int64_t)(done + padded) * MODEL_KV_ROWS);
                        snprintf(name, sizeof(name), "layer_%02d_vCache", L + 1);
                        dumpBufferFp16(g, st->vCache[L], name, (int64_t)(done + padded) * MODEL_KV_ROWS);
                    } else {
                        int64_t nt = (int64_t)(done + padded);
                        snprintf(name, sizeof(name), "layer_%02d_kCacheU8", L + 1);
                        dumpBufferU8(g, st->kCache[L], name, nt * MODEL_KV_ROWS);
                        snprintf(name, sizeof(name), "layer_%02d_vCacheU8", L + 1);
                        dumpBufferU8(g, st->vCache[L], name, nt * MODEL_KV_ROWS);
                        snprintf(name, sizeof(name), "layer_%02d_kScale", L + 1);
                        dumpBuffer(g, st->kScale[L], name, (int64_t)MODEL_KV_HEADS * MODEL_MAX_CTX);
                        snprintf(name, sizeof(name), "layer_%02d_kZero", L + 1);
                        dumpBuffer(g, st->kZero[L], name, (int64_t)MODEL_KV_HEADS * MODEL_MAX_CTX);
                        snprintf(name, sizeof(name), "layer_%02d_vScale", L + 1);
                        dumpBuffer(g, st->vScale[L], name, (int64_t)MODEL_KV_HEADS * MODEL_MAX_CTX);
                        snprintf(name, sizeof(name), "layer_%02d_vZero", L + 1);
                        dumpBuffer(g, st->vZero[L], name, (int64_t)MODEL_KV_HEADS * MODEL_MAX_CTX);
                    }
                }
                snprintf(name, sizeof(name), "layer_%02d_act", L + 1);
                dumpBuffer(g, st->act, name, (int64_t)padded * MODEL_FFN_N);
                snprintf(name, sizeof(name), "layer_%02d_gAct", L + 1);
                dumpBuffer(g, st->gAct, name, (int64_t)padded * MODEL_FFN_N);
                snprintf(name, sizeof(name), "layer_%02d_uAct", L + 1);
                dumpBuffer(g, st->uAct, name, (int64_t)padded * MODEL_FFN_N);
                snprintf(name, sizeof(name), "layer_%02d_invRms", L + 1);
                dumpBuffer(g, st->invRms, name, (int64_t)padded);
                if (g->spec->layers[L].attn.type == ATTENTION_DELTA) {
                    snprintf(name, sizeof(name), "layer_%02d_yGated", L + 1);
                    dumpBuffer(g, st->yGated, name, (int64_t)padded * MODEL_K);
                }
            }
            for (int L = g->dumpLayers; L < g->spec->layerCount; L++) {
                int n = 0;
                buildLayer(g, g->prefillOps, &n, L, 1, padded, 0, done + padded, done);
                executeChunked(g->s, g->prefillOps, n, "prefill", (int)(g->nextPos + done));
            }
        } else {
            g->prefillOpCount = compilePrefill(g, padded, done);
            executeChunked(g->s, g->prefillOps, g->prefillOpCount, "prefill", (int)(g->nextPos + done));
        }
        done += cur;
        lastCur = cur;
    }

    float* hCpu = (float*)malloc(sizeof(float) * m * MODEL_K);
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->h, hCpu);
    memcpy(st->lastRow.mappedMemory, hCpu + (lastCur - 1) * MODEL_K, sizeof(float) * MODEL_K);
    free(hCpu);

    stateSetPosition(g->s, &g->st, g->nextPos + (uint32_t)nTokens);

    executeLogged(g->s, g->finalOps, g->finalOpCount, "prefill", (int)g->nextPos);

    g->nextPos += (uint32_t)nTokens;

    uint32_t results[4] = {0};
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->result, results);
    return results[0];
}

void generateTokens(generator* g, const uint32_t* prompt, int nPrompt, int maxNewTokens, void (*emit)(uint32_t token, void* ctx), void* ctx) {
    int limit = g->maxCtx - nPrompt;
    if (limit < 1) limit = 1;
    if (maxNewTokens > limit) maxNewTokens = limit;

    uint32_t token = runPrefill(g, prompt, nPrompt);
    int count = 1;
    emit(token, ctx);
    if (token == (uint32_t)g->eos || maxNewTokens <= 1) {
        return;
    }

    int decodeTokens = maxNewTokens - 1;
    int fullGroups = decodeTokens / DECODE_GROUP;
    int rem = decodeTokens % DECODE_GROUP;

    int curSplit = (int)(g->nextPos >= ATT_SPLIT_THRESHOLD);
    operation* curOps = curSplit ? g->groupOps : g->groupOpsShort;
    int curCount = curSplit ? g->groupOpCount : g->groupOpCountShort;

    ((uint32_t*)g->st.tokenIds.mappedMemory)[0] = token;
    int dumpFirst = (g->dumpDir[0] != '\0');
    executeRecord(&g->s, curOps, dumpFirst ? curCount / DECODE_GROUP : curCount);
    executeSubmitNow(&g->s);
    if (dumpFirst) {
        executeWaitLast(&g->s);
        dumpBuffer(g, g->st.h, "decode_h", MODEL_K);
        dumpBuffer(g, g->st.embOut, "decode_emb", MODEL_K);
        executeRecord(&g->s, curOps + curCount / DECODE_GROUP, curCount - curCount / DECODE_GROUP);
        executeSubmitNow(&g->s);
    }

    int eos = 0;
    int units = fullGroups + (rem > 0 ? 1 : 0);
    double prevDelay = FIRST_TOKEN_DELAY_MS;
    for (int u = 0; u < units && !eos; u++) {
        int cur = (u == fullGroups) ? rem : DECODE_GROUP;
        int next = ((rem > 0) && (u + 1 == fullGroups)) ? rem : DECODE_GROUP;

        int nextSplit = (int)((g->nextPos + cur) >= ATT_SPLIT_THRESHOLD);
        operation* nextOps = nextSplit ? g->groupOps : g->groupOpsShort;
        int nextCount = nextSplit ? g->groupOpCount : g->groupOpCountShort;

        uint32_t tokens[DECODE_GROUP];
        double t0 = nowMs();
        executeRecord(&g->s, nextOps, next * (nextCount / DECODE_GROUP));
        executeWaitLast(&g->s);
        logLastFrame(&g->s, curOps, cur * (curCount / DECODE_GROUP), "decode", (int)g->nextPos);
        readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, g->st.result, tokens);
        double t1 = nowMs();
        if (isTimingEnabled()) {
            fprintf(stderr, "[decode][tok=%5d] group=%d  %7.3f ms/tok\n", (int)g->nextPos, cur, (t1 - t0) / (double)cur);
        }
        ((uint32_t*)g->st.tokenIds.mappedMemory)[0] = tokens[cur - 1];
        executeSubmitNow(&g->s);
        g->nextPos += cur;

        for (int j = 0; j < cur && count < maxNewTokens; j++) {
            emit(tokens[j], ctx);
            count++;
            if (tokens[j] == (uint32_t)g->eos) {
                eos = 1;
                break;
            }
            sleepMs(prevDelay);
        }

        prevDelay = (t1 - t0) / (double)cur;

        curSplit = nextSplit;
        curOps = nextOps;
        curCount = nextCount;
    }
    executeWaitLast(&g->s);
}

void resetGenerator(generator* g) {
    int count = 0;
    buffer states[MODEL_LAYERS * 2 + 1];
    for (int L = 0; L < g->spec->layerCount; L++) {
        if (g->spec->layers[L].attn.type == ATTENTION_DELTA) {
            states[count++] = g->st.stateS[L];
            states[count++] = g->st.convHist[L];
        }
    }
    states[count++] = g->st.sampleHistory;
    createTransferAndCopy(g->s.dev.device, g->s.dev.queue, states, count);
    g->nextPos = 0;
}

void generatorSetSampling(generator* g, const sample_params* p, uint32_t seed) {
    if (g->st.sampleParams.mappedMemory != NULL) {
        memcpy(g->st.sampleParams.mappedMemory, p, sizeof(sample_params));
    }
    if (g->st.sampleRng.mappedMemory != NULL) {
        *(uint32_t*)g->st.sampleRng.mappedMemory = seed;
    }
    int sampling = (p->temperature > 0.0f) ? 1 : 0;
    if (g->sampling == sampling) return;
    g->sampling = sampling;
    g->groupOpCount = compileDecodeGroup(g, g->groupOps, 1);
    g->groupOpCountShort = compileDecodeGroup(g, g->groupOpsShort, 0);
    g->finalOpCount = compileFinal(g);
}

void generatorSetDumpDir(generator* g, const char* dir) {
    if (dir == NULL) {
        g->dumpDir[0] = '\0';
        return;
    }
    snprintf(g->dumpDir, sizeof(g->dumpDir), "%s", dir);
}

void generatorDumpPrefill(generator* g, int rows) {
    if (g->dumpDir[0] == '\0') return;
    model_state* st = &g->st;
    dumpBuffer(g, st->embOut, "embOut", (int64_t)rows * MODEL_K);
    dumpBuffer(g, st->h, "h", (int64_t)rows * MODEL_K);
    dumpBuffer(g, st->lastRow, "lastRow", MODEL_K);
}

void generatorSetDumpLayers(generator* g, int layers) {
    g->dumpLayers = layers;
}

void generatorDumpDecodeStep(generator* g, int step) {
    if (g->dumpDir[0] == '\0') return;
    model_state* st = &g->st;
    char name[64];
    snprintf(name, sizeof(name), "decode_%02d_h", step);
    dumpBuffer(g, st->h, name, MODEL_K);
    snprintf(name, sizeof(name), "decode_%02d_q", step);
    dumpBuffer(g, st->qOut, name, MODEL_K);
}
