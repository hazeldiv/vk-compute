#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "session.h"
#include "buffer.h"
#include "dispatch.h"
#include "data.h"
#include "validation.h"

static void validate_softmax_v(const float* x, const float* v, float* o, int n) {
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;
    float acc[256] = {0};

    for (int k = 0; k < n; k += 256) {
        int tile_size = (n - k < 256) ? (n - k) : 256;

        float tile_max = -FLT_MAX;
        for (int i = 0; i < tile_size; i++) {
            if (x[k + i] > tile_max) tile_max = x[k + i];
        }

        float exp_vals[256];
        float tile_sum = 0.0f;
        for (int i = 0; i < tile_size; i++) {
            exp_vals[i] = expf(x[k + i] - tile_max);
            tile_sum += exp_vals[i];
        }

        if (tile_sum > 0.0f) {
            float new_max = fmaxf(max_val, tile_max);
            float scale_prev = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
            float scale_tile = expf(tile_max - new_max);

            sum_val = sum_val * scale_prev + tile_sum * scale_tile;

            for (int col = 0; col < 256; col++) {
                float tile_acc = 0.0f;
                for (int i = 0; i < tile_size; i++) {
                    tile_acc += exp_vals[i] * v[(k + i) * 256 + col];
                }
                acc[col] = acc[col] * scale_prev + tile_acc * scale_tile;
            }
            max_val = new_max;
        }
    }

    for (int col = 0; col < 256; col++) {
        o[col] = (sum_val > 0.0f) ? acc[col] / sum_val : 0.0f;
    }
}

static void validate_attention(const float* q, const float* k, const float* v, float* o, int seq, int heads, int kv_heads, int dim) {
    for (int h = 0; h < heads; h++) {
        int kv = h / (heads / kv_heads);
        const float* qh = q + h * dim;
        const float* kh = k + kv * dim * seq;
        const float* vh = v + kv * dim * seq;

        float max_val = -FLT_MAX;
        float sum_val = 0.0f;
        float acc[256] = {0};

        for (int t = 0; t < seq; t += 256) {
            int tile = (seq - t < 256) ? (seq - t) : 256;

            float scores[256];
            float tile_max = -FLT_MAX;
            for (int i = 0; i < tile; i++) {
                float sc = 0.0f;
                for (int d = 0; d < dim; d++) sc += qh[d] * kh[d * seq + (t + i)];
                scores[i] = sc;
                if (sc > tile_max) tile_max = sc;
            }

            float expv[256];
            float tile_sum = 0.0f;
            for (int i = 0; i < tile; i++) {
                expv[i] = expf(scores[i] - tile_max);
                tile_sum += expv[i];
            }

            float new_max = fmaxf(max_val, tile_max);
            float scale_prev = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
            float scale_tile = expf(tile_max - new_max);
            sum_val = sum_val * scale_prev + tile_sum * scale_tile;

            for (int d = 0; d < dim; d++) {
                float tile_acc = 0.0f;
                for (int i = 0; i < tile; i++) tile_acc += expv[i] * vh[d * seq + (t + i)];
                acc[d] = acc[d] * scale_prev + tile_acc * scale_tile;
            }
            max_val = new_max;
        }

        for (int d = 0; d < dim; d++) o[h * dim + d] = (sum_val > 0.0f) ? acc[d] / sum_val : 0.0f;
    }
}

static double run_ops(session s, operation ops[], int n) {
    execute(s, ops, n);
    return getExecutionTime(s);
}

static void destroy_buffers(session s, buffer bufs[], int n) {
    for (int i = 0; i < n; i++) destroyBuffer(s.dev.device, bufs[i]);
}

static void report(const char* name, int idx, const float* out, const float* ref, int count, double ms) {
    float err = 0.0f;
    for (int i = 0; i < count; i++) {
        float e = fabsf(out[i] - ref[i]);
        if (e > err) err = e;
    }
    printf("%s: shader[%d]= %f ref[%d]= %f max_err= %f\n", name, idx, out[idx], idx, ref[idx], err);
    printf("%s time: %.3f ms\n", name, ms);
}

static void rms_norm_apply(const float* x, const float* g, float* xn, int k) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += x[i] * x[i];
    float rms = sqrtf(sum / (float)k + 1e-5f);
    float inv = 1.0f / rms;
    for (int i = 0; i < k; i++) xn[i] = x[i] * g[i] * inv;
}

static void gemv_ref_f32(const float* x, const float* w, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += x[i] * w[i * n + j];
        o[j] = acc;
    }
}

static void gemv_ref_fp16(const float* x, const uint16_t* w, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += x[i] * fp16_to_float(w[i * n + j]);
        o[j] = acc;
    }
}

static void gemv_ref_int8(const float* x, const QuantizedData* q, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        int bj = j / q->group_size;
        for (int i = 0; i < k; i++) {
            float dq = (float)q->data[i * n + j] * q->scale[bj * k + i] - q->z[bj * k + i];
            acc += x[i] * dq;
        }
        o[j] = acc;
    }
}

static void gemv_ref_int4(const float* x, const QuantizedData* q, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        int bj = j / q->group_size;
        for (int i = 0; i < k; i++) {
            uint8_t b = q->data[i * (n / 2) + j / 2];
            int nib = (j & 1) ? (b & 0x0F) : (b >> 4);
            float dq = (float)nib * q->scale[bj * k + i] - q->z[bj * k + i];
            acc += x[i] * dq;
        }
        o[j] = acc;
    }
}

static void swiglu_ref_f32(const float* x, const float* g, const float* wg, const float* wu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int i = 0; i < k; i++) {
            gate += xn[i] * wg[i * n + j];
            up += xn[i] * wu[i * n + j];
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

static void swiglu_ref_fp16(const float* x, const float* g, const uint16_t* wg, const uint16_t* wu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int i = 0; i < k; i++) {
            gate += xn[i] * fp16_to_float(wg[i * n + j]);
            up += xn[i] * fp16_to_float(wu[i * n + j]);
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

static void swiglu_ref_int8(const float* x, const float* g, const QuantizedData* qg, const QuantizedData* qu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        int bj = j / qg->group_size;
        for (int i = 0; i < k; i++) {
            float dg = (float)qg->data[i * n + j] * qg->scale[bj * k + i] - qg->z[bj * k + i];
            float du = (float)qu->data[i * n + j] * qu->scale[bj * k + i] - qu->z[bj * k + i];
            gate += xn[i] * dg;
            up += xn[i] * du;
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

static void swiglu_ref_int4(const float* x, const float* g, const QuantizedData* qg, const QuantizedData* qu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        int bj = j / qg->group_size;
        for (int i = 0; i < k; i++) {
            uint8_t bg = qg->data[i * (n / 2) + j / 2];
            uint8_t bu = qu->data[i * (n / 2) + j / 2];
            int ng = (j & 1) ? (bg & 0x0F) : (bg >> 4);
            int nu = (j & 1) ? (bu & 0x0F) : (bu >> 4);
            float dg = (float)ng * qg->scale[bj * k + i] - qg->z[bj * k + i];
            float du = (float)nu * qu->scale[bj * k + i] - qu->z[bj * k + i];
            gate += xn[i] * dg;
            up += xn[i] * du;
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

static void dequant_attention_int8(const QuantizedData* kq, const QuantizedData* vq, float* kf, float* vf, int rows, int seq) {
    for (int i = 0; i < rows * seq; i++) {
        int row = i / seq;
        int tok = i % seq;
        int s_idx = (tok / kq->group_size) * rows + row;
        kf[i] = (float)kq->data[i] * kq->scale[s_idx] - kq->z[s_idx];
        vf[i] = (float)vq->data[i] * vq->scale[s_idx] - vq->z[s_idx];
    }
}

static void dequant_attention_int4(const QuantizedData* kq, const QuantizedData* vq, float* kf, float* vf, int rows, int seq) {
    for (int i = 0; i < rows * seq; i++) {
        int row = i / seq;
        int tok = i % seq;
        int s_idx = (tok / kq->group_size) * rows + row;
        uint8_t kb = kq->data[i / 2];
        uint8_t vb = vq->data[i / 2];
        int kn = (i & 1) ? (kb & 0x0F) : (kb >> 4);
        int vn = (i & 1) ? (vb & 0x0F) : (vb >> 4);
        kf[i] = (float)kn * kq->scale[s_idx] - kq->z[s_idx];
        vf[i] = (float)vn * vq->scale[s_idx] - vq->z[s_idx];
    }
}

void validateGEMV(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    float* transposed = (float*)malloc(sizeof(float) * K * N);
    transpose_block16((uint8_t*)d->weight, (uint8_t*)transposed, K, N, QUANT_FP32);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(float) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV.spv", .buffers = {inputBuffer, weightBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_f32(d->input, d->weight, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMVFP16(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)d->weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(d->input, d->weightFP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMVINT8(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(d->weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT8.scale, sizeof(float) * K * N / d->weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT8.z, sizeof(float) * K * N / d->weightINT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(d->input, &d->weightINT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateGEMVINT4(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(d->weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT4.scale, sizeof(float) * K * N / d->weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT4.z, sizeof(float) * K * N / d->weightINT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(d->input, &d->weightINT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormGEMVFP16(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)d->weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-GEMV-FP16.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(d->input, d->gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(xn, d->weightFP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
    free(xn);
}

void validateRmsNormGEMVINT8(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(d->weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT8.scale, sizeof(float) * K * N / d->weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT8.z, sizeof(float) * K * N / d->weightINT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-GEMV-INT8.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(d->input, d->gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(xn, &d->weightINT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
    free(xn);
}

void validateRmsNormGEMVINT4(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(d->weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT4.scale, sizeof(float) * K * N / d->weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT4.z, sizeof(float) * K * N / d->weightINT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-GEMV-INT4.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(d->input, d->gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(xn, &d->weightINT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
    free(xn);
}

void validateRmsNormSwigluFfn(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    float* transposed = (float*)malloc(sizeof(float) * K * N);
    transpose_block16((uint8_t*)d->weight, (uint8_t*)transposed, K, N, QUANT_FP32);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(float) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, weightBuffer, outBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_f32(d->input, d->gamma, d->weight, d->weight, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnFP16(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* transposed2 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)d->weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)d->weight2FP16, (uint8_t*)transposed2, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-FP16.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_fp16(d->input, d->gamma, d->weightFP16, d->weight2FP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnINT8(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(d->weightINT8.data, transposed, K, N, QUANT_INT8);
    transpose_block16(d->weight2INT8.data, transposed2, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT8.scale, sizeof(float) * K * N / d->weightINT8.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT8.z, sizeof(float) * K * N / d->weightINT8.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->weight2INT8.scale, sizeof(float) * K * N / d->weight2INT8.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->weight2INT8.z, sizeof(float) * K * N / d->weight2INT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-INT8.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_int8(d->input, d->gamma, &d->weightINT8, &d->weight2INT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnINT4(session s, const ValidationData* d) {
    int M = d->M;
    int N = d->N;
    int K = d->K;
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(d->weightINT4.data, transposed, K, N, QUANT_INT4);
    transpose_block16(d->weight2INT4.data, transposed2, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT4.scale, sizeof(float) * K * N / d->weightINT4.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->weightINT4.z, sizeof(float) * K * N / d->weightINT4.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->weight2INT4.scale, sizeof(float) * K * N / d->weight2INT4.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->weight2INT4.z, sizeof(float) * K * N / d->weight2INT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-INT4.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_int4(d->input, d->gamma, &d->weightINT4, &d->weight2INT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateOnlineSoftmax(session s, const ValidationData* d) {
    int n = d->softmax_n;
    float* out = (float*)calloc(256, sizeof(float));

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->softmax_x, sizeof(float) * n, MEMORY_VRAM);
    buffer vBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->softmax_v, sizeof(float) * n * 256, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * 256, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, vBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);

    operation ops[] = {
        {.shader = "online-softmax.spv", .buffers = {xBuffer, vBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {n}, .pushConstantCount = 1,
         .dispatchX = 1, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * 256);
    validate_softmax_v(d->softmax_x, d->softmax_v, ref, n);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("online-softmax", 40, out, ref, 256, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateAttentionFP16(session s, const ValidationData* d) {
    int seq = d->att_seq;
    int heads = d->att_heads;
    int kv_heads = d->att_kv_heads;
    int dim = d->att_dim;
    int kvs = kv_heads * dim * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) {
        kf[i] = fp16_to_float(d->att_k[i]);
        vf[i] = fp16_to_float(d->att_v[i]);
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_k, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_v, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);

    operation ops[] = {
        {.shader = "Att-full-FP16.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(d->att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention FP16", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 4);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionINT8(session s, const ValidationData* d) {
    int seq = d->att_seq;
    int heads = d->att_heads;
    int kv_heads = d->att_kv_heads;
    int dim = d->att_dim;
    int rows = kv_heads * dim;
    int blocks = seq / 256;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    dequant_attention_int8(&d->att_k_i8, &d->att_v_i8, kf, vf, rows, seq);
    float* out = (float*)calloc(heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_k_i8.data, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_v_i8.data, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_k_i8.scale, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer kZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_k_i8.z, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer vScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_v_i8.scale, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer vZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_v_i8.z, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScale, kZero, vScale, vZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);

    operation ops[] = {
        {.shader = "Att-full-INT8.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScale, kZero, vScale, vZero}, .bufferCount = 8,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(d->att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention INT8", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 8);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionINT4(session s, const ValidationData* d) {
    int seq = d->att_seq;
    int heads = d->att_heads;
    int kv_heads = d->att_kv_heads;
    int dim = d->att_dim;
    int rows = kv_heads * dim;
    int blocks = seq / 256;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    dequant_attention_int4(&d->att_k_i4, &d->att_v_i4, kf, vf, rows, seq);
    float* out = (float*)calloc(heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_k_i4.data, sizeof(uint8_t) * kvs / 2, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_v_i4.data, sizeof(uint8_t) * kvs / 2, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_k_i4.scale, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer kZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_k_i4.z, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer vScale = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_v_i4.scale, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer vZero = createBuffer(s.dev.device, s.dev.physicalDevice, d->att_v_i4.z, sizeof(float) * rows * blocks, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScale, kZero, vScale, vZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);

    operation ops[] = {
        {.shader = "Att-full-INT4.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScale, kZero, vScale, vZero}, .bufferCount = 8,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(d->att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention INT4", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 8);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}
