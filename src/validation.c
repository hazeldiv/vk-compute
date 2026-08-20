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

static void validate_attention_multi(const float* q, const float* k, const float* v, float* o, int seq, int heads, int kv_heads, int dim) {
    for (int m = 0; m < seq; m++)
        for (int h = 0; h < heads; h++) {
            int kv = h / (heads / kv_heads);
            const float* qh = q + (m * heads + h) * dim;
            const float* kh = k + kv * dim * seq;
            const float* vh = v + kv * dim * seq;
            float max_val = -FLT_MAX, sum_val = 0.0f;
            float acc[256] = {0};
            for (int t = 0; t <= m; t += 256) {
                int tile = ((m - t + 1) < 256) ? (m - t + 1) : 256;
                float scores[256]; float tile_max = -FLT_MAX;
                for (int i = 0; i < tile; i++) {
                    float sc = 0.0f;
                    for (int d = 0; d < dim; d++) sc += qh[d] * kh[d * seq + (t + i)];
                    scores[i] = sc; if (sc > tile_max) tile_max = sc;
                }
                float expv[256]; float tile_sum = 0.0f;
                for (int i = 0; i < tile; i++) { expv[i] = expf(scores[i] - tile_max); tile_sum += expv[i]; }
                float new_max = fmaxf(max_val, tile_max);
                float sp = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
                float st = expf(tile_max - new_max);
                sum_val = sum_val * sp + tile_sum * st;
                for (int d = 0; d < dim; d++) {
                    float ta = 0.0f;
                    for (int i = 0; i < tile; i++) ta += expv[i] * vh[d * seq + (t + i)];
                    acc[d] = acc[d] * sp + ta * st;
                }
                max_val = new_max;
            }
            for (int d = 0; d < dim; d++) o[(m * heads + h) * dim + d] = (sum_val > 0.0f) ? acc[d] / sum_val : 0.0f;
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

static void gemm_ref_fp16(const float* x, const uint16_t* w, float* o, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) acc += x[i * k + t] * fp16_to_float(w[t * n + j]);
            o[i * n + j] = acc;
        }
}

static void gemm_ref_int8(const float* x, const QuantizedData* q, float* o, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            int bj = j / q->group_size;
            for (int t = 0; t < k; t++) {
                float dq = (float)q->data[t * n + j] * q->scale[bj * k + t] - q->z[bj * k + t];
                acc += x[i * k + t] * dq;
            }
            o[i * n + j] = acc;
        }
}

static void gemm_ref_int4(const float* x, const QuantizedData* q, float* o, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            int bj = j / q->group_size;
            for (int t = 0; t < k; t++) {
                uint8_t b = q->data[t * (n / 2) + j / 2];
                int nib = (j & 1) ? (b & 0x0F) : (b >> 4);
                acc += x[i * k + t] * ((float)nib * q->scale[bj * k + t] - q->z[bj * k + t]);
            }
            o[i * n + j] = acc;
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

void validateGEMV(session s, int M, int N, int K, float* input, float* weight) {
    float* out = (float*)calloc(M * N, sizeof(float));

    float* transposed = (float*)malloc(sizeof(float) * K * N);
    transpose_block16((uint8_t*)weight, (uint8_t*)transposed, K, N, QUANT_FP32);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
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
    gemv_ref_f32(input, weight, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMVFP16(session s, int M, int N, int K, float* input, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
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
    gemv_ref_fp16(input, weightFP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMVINT8(session s, int M, int N, int K, float* input, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
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
    gemv_ref_int8(input, &weightINT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateGEMVINT4(session s, int M, int N, int K, float* input, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
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
    gemv_ref_int4(input, &weightINT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateGEMMFP16(session s, int M, int N, int K, float* input, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_fp16(input, weightFP16, ref, M, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMMINT8(session s, int M, int N, int K, float* input, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int8(input, &weightINT8, ref, M, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateGEMMINT4(session s, int M, int N, int K, float* input, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int4(input, &weightINT4, ref, M, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormGEMVFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
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
    rms_norm_apply(input, gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(xn, weightFP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
    free(xn);
}

void validateRmsNormGEMVINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
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
    rms_norm_apply(input, gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(xn, &weightINT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
    free(xn);
}

void validateRmsNormGEMVINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
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
    rms_norm_apply(input, gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(xn, &weightINT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
    free(xn);
}

void validateGemvAddFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-ADD-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, residualBuffer}, .bufferCount = 4,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(input, weightFP16, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-ADD FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
}

void validateGemvAddINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-ADD-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(input, &weightINT8, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-ADD INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateGemvAddINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-ADD-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(input, &weightINT4, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-ADD INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfn(session s, int M, int N, int K, float* input, float* gamma, float* weight) {
    float* out = (float*)calloc(M * N, sizeof(float));

    float* transposed = (float*)malloc(sizeof(float) * K * N);
    transpose_block16((uint8_t*)weight, (uint8_t*)transposed, K, N, QUANT_FP32);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
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
    swiglu_ref_f32(input, gamma, weight, weight, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* transposed2 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)weight2FP16, (uint8_t*)transposed2, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
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
    swiglu_ref_fp16(input, gamma, weightFP16, weight2FP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    transpose_block16(weight2INT8.data, transposed2, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.scale, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.z, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
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
    swiglu_ref_int8(input, gamma, &weightINT8, &weight2INT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    transpose_block16(weight2INT4.data, transposed2, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.scale, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.z, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
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
    swiglu_ref_int4(input, gamma, &weightINT4, &weight2INT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateOnlineSoftmax(session s, int softmax_n, float* softmax_x, float* softmax_v) {
    int n = softmax_n;
    float* out = (float*)calloc(256, sizeof(float));

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, softmax_x, sizeof(float) * n, MEMORY_VRAM);
    buffer vBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, softmax_v, sizeof(float) * n * 256, MEMORY_VRAM);
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
    validate_softmax_v(softmax_x, softmax_v, ref, n);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("online-softmax", 40, out, ref, 256, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateAttentionFP16(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) {
        kf[i] = fp16_to_float(att_k[i]);
        vf[i] = fp16_to_float(att_v[i]);
    }
    uint16_t* att_k_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++) {
        for (int r = 0; r < rows; r++) {
            att_k_t[s2 * rows + r] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(att_k_t);
    free(att_v_t);

    operation ops[] = {
        {.shader = "Att-full-FP16.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention FP16", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 4);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionINT8(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) {
        kf[i] = fp16_to_float(att_k[i]);
        vf[i] = fp16_to_float(att_v[i]);
    }
    uint16_t* att_k_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++) {
        for (int r = 0; r < rows; r++) {
            att_k_t[s2 * rows + r] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(att_k_t);
    free(att_v_t);

    operation ops[] = {
        {.shader = "Att-full-INT8.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention INT8", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 4);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionINT4(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * seq + s2] = ks;
            kZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * seq + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * seq + s2] = vs;
            vZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * seq + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * seq + s2] - kZero[h * seq + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * seq + s2] - vZero[h * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer kZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);

    operation ops[] = {
        {.shader = "Att-full-INT4.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf}, .bufferCount = 8,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention INT4", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 8);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionGEMMFP16(session s, int seq, int heads, int kv_heads, int dim) {
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* att_q = getData(7777, seq, heads * dim);
    uint16_t* att_k = getDataFP16(8100, rows, seq);
    uint16_t* att_v = getDataFP16(9100, rows, seq);

    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) { kf[i] = fp16_to_float(att_k[i]); vf[i] = fp16_to_float(att_v[i]); }

    uint16_t* att_k_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++)
        for (int r = 0; r < rows; r++) {
            att_k_t[s2 * rows + r] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }

    float* out = (float*)calloc(seq * heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);

    operation ops[] = {
        {.shader = "Att-full-GEMM-FP16.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = seq / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * seq * heads * dim);
    validate_attention_multi(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention-GEMM FP16", 100, out, ref, seq * heads * dim, ms);

    destroy_buffers(s, bufs, 4);
    free(att_q); free(att_k); free(att_v);
    free(kf); free(vf); free(att_k_t); free(att_v_t);
    free(out); free(ref);
}

void validateAttentionGEMMINT8(session s, int seq, int heads, int kv_heads, int dim) {
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* att_q = getData(7777, seq, heads * dim);
    uint16_t* att_k = getDataFP16(8100, rows, seq);
    uint16_t* att_v = getDataFP16(9100, rows, seq);

    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) { kf[i] = fp16_to_float(att_k[i]); vf[i] = fp16_to_float(att_v[i]); }

    uint16_t* att_k_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++)
        for (int r = 0; r < rows; r++) {
            att_k_t[s2 * rows + r] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }

    float* out = (float*)calloc(seq * heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);

    operation ops[] = {
        {.shader = "Att-full-GEMM-INT8.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = seq / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * seq * heads * dim);
    validate_attention_multi(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention-GEMM INT8", 100, out, ref, seq * heads * dim, ms);

    destroy_buffers(s, bufs, 4);
    free(att_q); free(att_k); free(att_v);
    free(kf); free(vf); free(att_k_t); free(att_v_t);
    free(out); free(ref);
}

void validateAttentionGEMMINT4(session s, int seq, int heads, int kv_heads, int dim) {
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* att_q = getData(7777, seq, heads * dim);
    uint16_t* att_k = getDataFP16(8100, rows, seq);
    uint16_t* att_v = getDataFP16(9100, rows, seq);

    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * seq + s2] = ks;
            kZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * seq + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * seq + s2] = vs;
            vZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * seq + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * seq + s2] - kZero[h * seq + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * seq + s2] - vZero[h * seq + s2];
        }
    }
    float* out = (float*)calloc(seq * heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer kScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer kZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);

    operation ops[] = {
        {.shader = "Att-full-GEMM-INT4.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf}, .bufferCount = 8,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = seq / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * seq * heads * dim);
    validate_attention_multi(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention-GEMM INT4", 100, out, ref, seq * heads * dim, ms);

    destroy_buffers(s, bufs, 8);
    free(att_q); free(att_k); free(att_v);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

static void qkv_rope_ref(const float* proj, const float* theta, float* qref, float* kref, float* vref,
                         int n_total, int k_offset, int v_offset, int dim, int token) {
    int num_heads = n_total / dim;
    float* hn = (float*)malloc(sizeof(float) * n_total);
    for (int h = 0; h < num_heads; h++) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += proj[h * dim + i] * proj[h * dim + i];
        float inv = 1.0f / (sqrtf(sum / (float)dim + 1e-5f));
        for (int i = 0; i < dim; i++) hn[h * dim + i] = proj[h * dim + i] * inv;
    }
    for (int j = 0; j < n_total; j++) {
        if (j >= v_offset) {
            vref[j - v_offset] = proj[j];
            continue;
        }
        int col = j % dim;
        int head = (j / dim) * dim;
        float val;
        if (col < dim / 2) {
            float a = hn[j];
            float b = hn[head + col + dim / 2];
            float ang = token * theta[col];
            val = a * cosf(ang) - b * sinf(ang);
        } else {
            float a = hn[head + col - dim / 2];
            float b = hn[j];
            float ang = token * theta[col - dim / 2];
            val = a * sinf(ang) + b * cosf(ang);
        }
        if (j < k_offset) qref[j] = val;
        else kref[j - k_offset] = val;
    }
    free(hn);
}



void validateQkvRopeFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int n_total = (heads + 2 * kv_heads) * dim;
    int k_offset = heads * dim;
    int v_offset = (heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_fp16(xn, qkv_weightFP16, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, n_total, k_offset, v_offset, dim, ctx);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint16_t* kCache = (uint16_t*)calloc(rows * seq_len, sizeof(uint16_t));
    uint16_t* vCache = (uint16_t*)calloc(rows * seq_len, sizeof(uint16_t));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * k * n_total);
    transpose_block16((uint8_t*)qkv_weightFP16, (uint8_t*)transposed, k, n_total, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * k * n_total, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (dim / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint16_t) * rows * seq_len, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint16_t) * rows * seq_len, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 7);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-FP16.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer}, .bufferCount = 7,
         .pushConstants = {1, n_total, k, ctx, k_offset, v_offset, seq_len}, .pushConstantCount = 7,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) kstored[r] = fp16_to_float(kCache[(seq_len - 1) * rows + r]);
    for (int r = 0; r < rows; r++) vstored[r] = fp16_to_float(vCache[(seq_len - 1) * rows + r]);
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }


    report("QKV-Rope FP16", 100, qOut, qref, heads * dim, ms);
    report("QKV-Rope-kv FP16", 100, kvOut, kvRef, 2 * rows, ms);

    destroy_buffers(s, bufs, 7);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

void validateQkvRopeINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int n_total = (heads + 2 * kv_heads) * dim;
    int k_offset = heads * dim;
    int v_offset = (heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    int scaleCount = k * n_total / qkv_weightINT8.group_size;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_int8(xn, &qkv_weightINT8, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, n_total, k_offset, v_offset, dim, ctx);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint16_t* kCache = (uint16_t*)calloc(rows * seq_len, sizeof(uint16_t));
    uint16_t* vCache = (uint16_t*)calloc(rows * seq_len, sizeof(uint16_t));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total);
    transpose_block16(qkv_weightINT8.data, transposed, k, n_total, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (dim / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint16_t) * rows * seq_len, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint16_t) * rows * seq_len, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-INT8.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer}, .bufferCount = 9,
         .pushConstants = {1, n_total, k, ctx, k_offset, v_offset, seq_len}, .pushConstantCount = 7,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) kstored[r] = fp16_to_float(kCache[(seq_len - 1) * rows + r]);
    for (int r = 0; r < rows; r++) vstored[r] = fp16_to_float(vCache[(seq_len - 1) * rows + r]);
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }

    report("QKV-Rope INT8", 100, qOut, qref, heads * dim, ms);
    report("QKV-Rope-kv INT8", 100, kvOut, kvRef, 2 * rows, ms);

    destroy_buffers(s, bufs, 9);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

void validateQkvRopeINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int n_total = (heads + 2 * kv_heads) * dim;
    int k_offset = heads * dim;
    int v_offset = (heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    int scaleCount = k * n_total / qkv_weightINT4.group_size;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_int4(xn, &qkv_weightINT4, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, n_total, k_offset, v_offset, dim, ctx);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * seq_len, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total / 2);
    transpose_block16(qkv_weightINT4.data, transposed, k, n_total, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total / 2, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (dim / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 13);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-INT4.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer}, .bufferCount = 13,
         .pushConstants = {1, n_total, k, ctx, k_offset, v_offset, seq_len}, .pushConstantCount = 7,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        kstored[r] = (float)kCache[(seq_len - 1) * rows + r] * kScale[h * seq_len + seq_len - 1] - kZero[h * seq_len + seq_len - 1];
        vstored[r] = (float)vCache[(seq_len - 1) * rows + r] * vScale[h * seq_len + seq_len - 1] - vZero[h * seq_len + seq_len - 1];
    }
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }

    report("QKV-Rope INT4", 100, qOut, qref, heads * dim, ms);
    report("QKV-Rope-kv INT4", 100, kvOut, kvRef, 2 * rows, ms);

    destroy_buffers(s, bufs, 13);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

static void deltanet_ref(const float* proj, float* S, float* ygated, int n_qk, int n_v, int dim) {
    const float* Q = proj;
    const float* K = proj + n_qk * dim;
    const float* V = proj + 2 * n_qk * dim;
    const float* G = proj + 2 * n_qk * dim + n_v * dim;
    const float* A = proj + 2 * n_qk * dim + 2 * n_v * dim;
    const float* B = A + n_qk;
    for (int h = 0; h < n_v; h++) {
        int qk = h / 2;
        float alpha = expf(-logf(1.0f + expf(A[qk])));
        float beta = 1.0f / (1.0f + expf(-B[qk]));
        float* Sh = S + h * dim * dim;
        const float* Kh = K + qk * dim;
        const float* Qh = Q + qk * dim;
        const float* Vh = V + h * dim;
        const float* Gh = G + h * dim;
        float* yh = ygated + h * dim;
        float delta[128];
        for (int i = 0; i < dim; i++) {
            float vhat = 0.0f;
            for (int j = 0; j < dim; j++) vhat += Sh[i * dim + j] * Kh[j];
            delta[i] = Vh[i] - vhat;
        }
        for (int i = 0; i < dim; i++) {
            float y = 0.0f;
            for (int j = 0; j < dim; j++) y += (alpha * Sh[i * dim + j] + beta * delta[i] * Kh[j]) * Qh[j];
            yh[i] = y * (Gh[i] / (1.0f + expf(-Gh[i])));
        }
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                Sh[i * dim + j] = alpha * Sh[i * dim + j] + beta * delta[i] * Kh[j];
            }
        }
    }
}

void validateGatedDeltaNetFP16(session s, int K, float* input, float* input2, float* gamma, uint16_t* w_inFP16, uint16_t* woFP16) {
    int proj_n = 12320;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;

    float* xn = (float*)malloc(sizeof(float) * K);
    float* xn2 = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    rms_norm_apply(input2, gamma, xn2, K);
    float* p1 = (float*)malloc(sizeof(float) * proj_n);
    float* p2 = (float*)malloc(sizeof(float) * proj_n);
    gemv_ref_fp16(xn, w_inFP16, p1, proj_n, K);
    gemv_ref_fp16(xn2, w_inFP16, p2, proj_n, K);
    float* S = (float*)calloc(smat, sizeof(float));
    float* yg1 = (float*)malloc(sizeof(float) * out_n);
    float* yg2 = (float*)malloc(sizeof(float) * out_n);
    deltanet_ref(p1, S, yg1, n_qk, n_v, dim);
    deltanet_ref(p2, S, yg2, n_qk, n_v, dim);
    float* ref = (float*)malloc(sizeof(float) * out_n);
    gemv_ref_fp16(yg2, woFP16, ref, out_n, K);

    float* qOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(n_v * dim, sizeof(float));
    float* gOut = (float*)calloc(n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(n_qk, sizeof(float));
    float* bOut = (float*)calloc(n_qk, sizeof(float));
    float* yGated = (float*)calloc(out_n, sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    uint16_t* twOut = (uint16_t*)malloc(sizeof(uint16_t) * K * out_n);
    transpose_block16((uint8_t*)w_inFP16, (uint8_t*)twIn, K, proj_n, QUANT_FP16);
    transpose_block16((uint8_t*)woFP16, (uint8_t*)twOut, K, out_n, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer x2Buffer = createBuffer(s.dev.device, s.dev.physicalDevice, input2, sizeof(float) * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint16_t) * K * out_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * n_qk, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * n_qk, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, x2Buffer, gammaBuffer, wInBuffer, wOutBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 14);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-FP16.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 9,
         .pushConstants = {1, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-FP16.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-LinearProj-FP16.spv", .buffers = {x2Buffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 9,
         .pushConstants = {1, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-FP16.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 6);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet FP16", 100, out, ref, out_n, ms);
    report("GatedDeltaNet-S FP16", 100, Sbuf, S, smat, ms);

    destroy_buffers(s, bufs, 14);
    free(xn);
    free(xn2);
    free(p1);
    free(p2);
    free(S);
    free(yg1);
    free(yg2);
    free(ref);
    free(qOut);
    free(kOut);
    free(vOut);
    free(gOut);
    free(aOut);
    free(bOut);
    free(yGated);
    free(out);
    free(Sbuf);
}

void validateGatedDeltaNetINT8(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT8, QuantizedData woINT8) {
    int proj_n = 12320;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    float* xn = (float*)malloc(sizeof(float) * K);
    float* xn2 = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    rms_norm_apply(input2, gamma, xn2, K);
    float* p1 = (float*)malloc(sizeof(float) * proj_n);
    float* p2 = (float*)malloc(sizeof(float) * proj_n);
    gemv_ref_int8(xn, &w_inINT8, p1, proj_n, K);
    gemv_ref_int8(xn2, &w_inINT8, p2, proj_n, K);
    float* S = (float*)calloc(smat, sizeof(float));
    float* yg1 = (float*)malloc(sizeof(float) * out_n);
    float* yg2 = (float*)malloc(sizeof(float) * out_n);
    deltanet_ref(p1, S, yg1, n_qk, n_v, dim);
    deltanet_ref(p2, S, yg2, n_qk, n_v, dim);
    float* ref = (float*)malloc(sizeof(float) * out_n);
    gemv_ref_int8(yg2, &woINT8, ref, out_n, K);

    float* qOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(n_v * dim, sizeof(float));
    float* gOut = (float*)calloc(n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(n_qk, sizeof(float));
    float* bOut = (float*)calloc(n_qk, sizeof(float));
    float* yGated = (float*)calloc(out_n, sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n);
    transpose_block16(w_inINT8.data, twIn, K, proj_n, QUANT_INT8);
    transpose_block16(woINT8.data, twOut, K, out_n, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer x2Buffer = createBuffer(s.dev.device, s.dev.physicalDevice, input2, sizeof(float) * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * n_qk, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * n_qk, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 18);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-INT8.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT8.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-LinearProj-INT8.spv", .buffers = {x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT8.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 6);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet INT8", 100, out, ref, out_n, ms);
    report("GatedDeltaNet-S INT8", 100, Sbuf, S, smat, ms);

    destroy_buffers(s, bufs, 18);
    free(xn);
    free(xn2);
    free(p1);
    free(p2);
    free(S);
    free(yg1);
    free(yg2);
    free(ref);
    free(qOut);
    free(kOut);
    free(vOut);
    free(gOut);
    free(aOut);
    free(bOut);
    free(yGated);
    free(out);
    free(Sbuf);
}

void validateGatedDeltaNetINT4(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT4, QuantizedData woINT4) {
    int proj_n = 12320;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    float* xn = (float*)malloc(sizeof(float) * K);
    float* xn2 = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    rms_norm_apply(input2, gamma, xn2, K);
    float* p1 = (float*)malloc(sizeof(float) * proj_n);
    float* p2 = (float*)malloc(sizeof(float) * proj_n);
    gemv_ref_int4(xn, &w_inINT4, p1, proj_n, K);
    gemv_ref_int4(xn2, &w_inINT4, p2, proj_n, K);
    float* S = (float*)calloc(smat, sizeof(float));
    float* yg1 = (float*)malloc(sizeof(float) * out_n);
    float* yg2 = (float*)malloc(sizeof(float) * out_n);
    deltanet_ref(p1, S, yg1, n_qk, n_v, dim);
    deltanet_ref(p2, S, yg2, n_qk, n_v, dim);
    float* ref = (float*)malloc(sizeof(float) * out_n);
    gemv_ref_int4(yg2, &woINT4, ref, out_n, K);

    float* qOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(n_v * dim, sizeof(float));
    float* gOut = (float*)calloc(n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(n_qk, sizeof(float));
    float* bOut = (float*)calloc(n_qk, sizeof(float));
    float* yGated = (float*)calloc(out_n, sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n / 2);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n / 2);
    transpose_block16(w_inINT4.data, twIn, K, proj_n, QUANT_INT4);
    transpose_block16(woINT4.data, twOut, K, out_n, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer x2Buffer = createBuffer(s.dev.device, s.dev.physicalDevice, input2, sizeof(float) * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n / 2, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n / 2, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * n_qk, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * n_qk, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 18);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-INT4.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT4.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-LinearProj-INT4.spv", .buffers = {x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT4.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 6);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet INT4", 100, out, ref, out_n, ms);
    report("GatedDeltaNet-S INT4", 100, Sbuf, S, smat, ms);

    destroy_buffers(s, bufs, 18);
    free(xn);
    free(xn2);
    free(p1);
    free(p2);
    free(S);
    free(yg1);
    free(yg2);
    free(ref);
    free(qOut);
    free(kOut);
    free(vOut);
    free(gOut);
    free(aOut);
    free(bOut);
    free(yGated);
    free(out);
    free(Sbuf);
}

void validateGatedDeltaNetGEMMFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16, uint16_t* woFP16) {
    int proj_n = 12320;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;

    const float wscale = 1.0f / 64.0f;
    uint16_t* w_in_scaled = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    for (int i = 0; i < K * proj_n; i++)
        w_in_scaled[i] = float_to_fp16(fp16_to_float(w_inFP16[i]) * wscale);

    float* S_ref = (float*)calloc(smat, sizeof(float));
    float* yg_all = (float*)malloc(sizeof(float) * M * out_n);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_fp16(xn, w_in_scaled, proj, proj_n, K);
        deltanet_ref(proj, S_ref, yg_all + m * out_n, n_qk, n_v, dim);
        free(xn);
        free(proj);
    }
    float* ref = (float*)malloc(sizeof(float) * M * out_n);
    gemm_ref_fp16(yg_all, woFP16, ref, M, out_n, K);

    float* qOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* gOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(M * n_qk, sizeof(float));
    float* bOut = (float*)calloc(M * n_qk, sizeof(float));
    float* yGated = (float*)calloc(M * out_n, sizeof(float));
    float* out = (float*)calloc(M * out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    uint16_t* twOut = (uint16_t*)malloc(sizeof(uint16_t) * K * out_n);
    transpose_block16((uint8_t*)w_in_scaled, (uint8_t*)twIn, K, proj_n, QUANT_FP16);
    transpose_block16((uint8_t*)woFP16, (uint8_t*)twOut, K, out_n, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint16_t) * K * out_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * n_qk, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * n_qk, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, wOutBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 13);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-FP16.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 9,
         .pushConstants = {M, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1},
        {.shader = "GatedDeltaNet-GEMM.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {M}, .pushConstantCount = 1,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMM-FP16.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet-GEMM FP16", 100, out, ref, M * out_n, ms);
    report("GatedDeltaNet-GEMM-S FP16", 100, Sbuf, S_ref, smat, ms);

    destroy_buffers(s, bufs, 13);
    free(w_in_scaled);
    free(S_ref); free(yg_all); free(ref);
    free(qOut); free(kOut); free(vOut); free(gOut); free(aOut); free(bOut);
    free(yGated); free(out); free(Sbuf);
}

void validateGatedDeltaNetGEMMINT8(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT8, QuantizedData woINT8) {
    int proj_n = 12320;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    const float wscale = 1.0f / 64.0f;
    QuantizedData w_in_scaled = w_inINT8;
    float* w_in_s8 = (float*)malloc(sizeof(float) * K * inBlocks);
    float* w_in_z8 = (float*)malloc(sizeof(float) * K * inBlocks);
    for (int i = 0; i < K * inBlocks; i++) {
        w_in_s8[i] = w_inINT8.scale[i] * wscale;
        w_in_z8[i] = w_inINT8.z[i] * wscale;
    }
    w_in_scaled.scale = w_in_s8;
    w_in_scaled.z = w_in_z8;

    float* S_ref = (float*)calloc(smat, sizeof(float));
    float* yg_all = (float*)malloc(sizeof(float) * M * out_n);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int8(xn, &w_in_scaled, proj, proj_n, K);
        deltanet_ref(proj, S_ref, yg_all + m * out_n, n_qk, n_v, dim);
        free(xn);
        free(proj);
    }
    float* ref = (float*)malloc(sizeof(float) * M * out_n);
    gemm_ref_int8(yg_all, &woINT8, ref, M, out_n, K);

    float* qOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* gOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(M * n_qk, sizeof(float));
    float* bOut = (float*)calloc(M * n_qk, sizeof(float));
    float* yGated = (float*)calloc(M * out_n, sizeof(float));
    float* out = (float*)calloc(M * out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n);
    transpose_block16(w_inINT8.data, twIn, K, proj_n, QUANT_INT8);
    transpose_block16(woINT8.data, twOut, K, out_n, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * n_qk, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * n_qk, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 17);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT8.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 11,
         .pushConstants = {M, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1},
        {.shader = "GatedDeltaNet-GEMM.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {M}, .pushConstantCount = 1,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMM-INT8.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {M, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet-GEMM INT8", 100, out, ref, M * out_n, ms);
    report("GatedDeltaNet-GEMM-S INT8", 100, Sbuf, S_ref, smat, ms);

    destroy_buffers(s, bufs, 17);
    free(w_in_s8); free(w_in_z8);
    free(S_ref); free(yg_all); free(ref);
    free(qOut); free(kOut); free(vOut); free(gOut); free(aOut); free(bOut);
    free(yGated); free(out); free(Sbuf);
}

void validateGatedDeltaNetGEMMINT4(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT4, QuantizedData woINT4) {
    int proj_n = 12320;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    const float wscale = 1.0f / 64.0f;
    QuantizedData w_in_scaled = w_inINT4;
    float* w_in_s4 = (float*)malloc(sizeof(float) * K * inBlocks);
    float* w_in_z4 = (float*)malloc(sizeof(float) * K * inBlocks);
    for (int i = 0; i < K * inBlocks; i++) {
        w_in_s4[i] = w_inINT4.scale[i] * wscale;
        w_in_z4[i] = w_inINT4.z[i] * wscale;
    }
    w_in_scaled.scale = w_in_s4;
    w_in_scaled.z = w_in_z4;

    float* S_ref = (float*)calloc(smat, sizeof(float));
    float* yg_all = (float*)malloc(sizeof(float) * M * out_n);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int4(xn, &w_in_scaled, proj, proj_n, K);
        deltanet_ref(proj, S_ref, yg_all + m * out_n, n_qk, n_v, dim);
        free(xn);
        free(proj);
    }
    float* ref = (float*)malloc(sizeof(float) * M * out_n);
    gemm_ref_int4(yg_all, &woINT4, ref, M, out_n, K);

    float* qOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* gOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(M * n_qk, sizeof(float));
    float* bOut = (float*)calloc(M * n_qk, sizeof(float));
    float* yGated = (float*)calloc(M * out_n, sizeof(float));
    float* out = (float*)calloc(M * out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n / 2);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n / 2);
    transpose_block16(w_inINT4.data, twIn, K, proj_n, QUANT_INT4);
    transpose_block16(woINT4.data, twOut, K, out_n, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n / 2, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n / 2, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * n_qk, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * n_qk, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 17);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT4.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 11,
         .pushConstants = {M, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1},
        {.shader = "GatedDeltaNet-GEMM.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 8,
         .pushConstants = {M}, .pushConstantCount = 1,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMM-INT4.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {M, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet-GEMM INT4", 100, out, ref, M * out_n, ms);
    report("GatedDeltaNet-GEMM-S INT4", 100, Sbuf, S_ref, smat, ms);

    destroy_buffers(s, bufs, 17);
    free(w_in_s4); free(w_in_z4);
    free(S_ref); free(yg_all); free(ref);
    free(qOut); free(kOut); free(vOut); free(gOut); free(aOut); free(bOut);
    free(yGated); free(out); free(Sbuf);
}

void validateGemmAddFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-ADD-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, residualBuffer}, .bufferCount = 4,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_fp16(input, weightFP16, ref, M, N, K);
    for (int i = 0; i < M * N; i++) ref[i] += residual[i];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-ADD FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
}

void validateGemmAddINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-ADD-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int8(input, &weightINT8, ref, M, N, K);
    for (int i = 0; i < M * N; i++) ref[i] += residual[i];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-ADD INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateGemmAddINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-ADD-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int4(input, &weightINT4, ref, M, N, K);
    for (int i = 0; i < M * N; i++) ref[i] += residual[i];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-ADD INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMMFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* transposed2 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)weight2FP16, (uint8_t*)transposed2, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-GEMM-FP16.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_fp16(input + m * K, gamma, weightFP16, weight2FP16, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMMINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    transpose_block16(weight2INT8.data, transposed2, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.scale, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.z, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-GEMM-INT8.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int8(input + m * K, gamma, &weightINT8, &weight2INT8, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMMINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    transpose_block16(weight2INT4.data, transposed2, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.scale, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.z, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-GEMM-INT4.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int4(input + m * K, gamma, &weightINT4, &weight2INT4, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateRmsNormLinearProjGEMMFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16) {
    int proj_n = 12320;
    float* qref = (float*)malloc(sizeof(float) * M * 2048);
    float* kref = (float*)malloc(sizeof(float) * M * 2048);
    float* vref = (float*)malloc(sizeof(float) * M * 4096);
    float* gref = (float*)malloc(sizeof(float) * M * 4096);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_fp16(xn, w_inFP16, proj, proj_n, K);
        memcpy(qref + m * 2048, proj, sizeof(float) * 2048);
        memcpy(kref + m * 2048, proj + 2048, sizeof(float) * 2048);
        memcpy(vref + m * 4096, proj + 4096, sizeof(float) * 4096);
        memcpy(gref + m * 4096, proj + 8192, sizeof(float) * 4096);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * 2048, sizeof(float));
    float* kOut = (float*)calloc(M * 2048, sizeof(float));
    float* vOut = (float*)calloc(M * 4096, sizeof(float));
    float* gOut = (float*)calloc(M * 4096, sizeof(float));
    float* aOut = (float*)calloc(M * 16, sizeof(float));
    float* bOut = (float*)calloc(M * 16, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    transpose_block16((uint8_t*)w_inFP16, (uint8_t*)twIn, K, proj_n, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * 16, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * 16, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(twIn);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-FP16.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 9,
         .pushConstants = {M, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("RmsNorm-LinearProj-GEMM-FP16", 100, qOut, qref, M * 2048, ms);
    report("RmsNorm-LinearProj-GEMM-v FP16", 100, vOut, vref, M * 4096, ms);
    report("RmsNorm-LinearProj-GEMM-g FP16", 100, gOut, gref, M * 4096, ms);

    destroy_buffers(s, bufs, 9);
    free(qref); free(kref); free(vref); free(gref);
    free(qOut); free(kOut); free(vOut); free(gOut); free(aOut); free(bOut);
}

void validateRmsNormLinearProjGEMMINT8(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT8) {
    int proj_n = 12320;
    int inBlocks = (proj_n + 255) / 256;
    float* qref = (float*)malloc(sizeof(float) * M * 2048);
    float* kref = (float*)malloc(sizeof(float) * M * 2048);
    float* vref = (float*)malloc(sizeof(float) * M * 4096);
    float* gref = (float*)malloc(sizeof(float) * M * 4096);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int8(xn, &w_inINT8, proj, proj_n, K);
        memcpy(qref + m * 2048, proj, sizeof(float) * 2048);
        memcpy(kref + m * 2048, proj + 2048, sizeof(float) * 2048);
        memcpy(vref + m * 4096, proj + 4096, sizeof(float) * 4096);
        memcpy(gref + m * 4096, proj + 8192, sizeof(float) * 4096);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * 2048, sizeof(float));
    float* kOut = (float*)calloc(M * 2048, sizeof(float));
    float* vOut = (float*)calloc(M * 4096, sizeof(float));
    float* gOut = (float*)calloc(M * 4096, sizeof(float));
    float* aOut = (float*)calloc(M * 16, sizeof(float));
    float* bOut = (float*)calloc(M * 16, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n);
    transpose_block16(w_inINT8.data, twIn, K, proj_n, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * 16, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * 16, MEMORY_VRAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 11);
    free(twIn);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT8.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 11,
         .pushConstants = {M, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("RmsNorm-LinearProj-GEMM-INT8", 100, qOut, qref, M * 2048, ms);
    report("RmsNorm-LinearProj-GEMM-v INT8", 100, vOut, vref, M * 4096, ms);
    report("RmsNorm-LinearProj-GEMM-g INT8", 100, gOut, gref, M * 4096, ms);

    destroy_buffers(s, bufs, 11);
    free(qref); free(kref); free(vref); free(gref);
    free(qOut); free(kOut); free(vOut); free(gOut); free(aOut); free(bOut);
}

void validateRmsNormLinearProjGEMMINT4(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT4) {
    int proj_n = 12320;
    int inBlocks = (proj_n + 255) / 256;
    float* qref = (float*)malloc(sizeof(float) * M * 2048);
    float* kref = (float*)malloc(sizeof(float) * M * 2048);
    float* vref = (float*)malloc(sizeof(float) * M * 4096);
    float* gref = (float*)malloc(sizeof(float) * M * 4096);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int4(xn, &w_inINT4, proj, proj_n, K);
        memcpy(qref + m * 2048, proj, sizeof(float) * 2048);
        memcpy(kref + m * 2048, proj + 2048, sizeof(float) * 2048);
        memcpy(vref + m * 4096, proj + 4096, sizeof(float) * 4096);
        memcpy(gref + m * 4096, proj + 8192, sizeof(float) * 4096);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * 2048, sizeof(float));
    float* kOut = (float*)calloc(M * 2048, sizeof(float));
    float* vOut = (float*)calloc(M * 4096, sizeof(float));
    float* gOut = (float*)calloc(M * 4096, sizeof(float));
    float* aOut = (float*)calloc(M * 16, sizeof(float));
    float* bOut = (float*)calloc(M * 16, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n / 2);
    transpose_block16(w_inINT4.data, twIn, K, proj_n, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n / 2, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * 16, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * 16, MEMORY_VRAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 11);
    free(twIn);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT4.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, gOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 11,
         .pushConstants = {M, proj_n, K}, .pushConstantCount = 3,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("RmsNorm-LinearProj-GEMM-INT4", 100, qOut, qref, M * 2048, ms);
    report("RmsNorm-LinearProj-GEMM-v INT4", 100, vOut, vref, M * 4096, ms);
    report("RmsNorm-LinearProj-GEMM-g INT4", 100, gOut, gref, M * 4096, ms);

    destroy_buffers(s, bufs, 11);
    free(qref); free(kref); free(vref); free(gref);
    free(qOut); free(kOut); free(vOut); free(gOut); free(aOut); free(bOut);
}

void validateQkvRopeGEMMFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta, int M) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int n_total = (heads + 2 * kv_heads) * dim;
    int k_offset = heads * dim;
    int v_offset = (heads + kv_heads) * dim;
    int rows = kv_heads * dim;

    float* qref = (float*)malloc(sizeof(float) * M * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * M * rows);
    float* vref = (float*)malloc(sizeof(float) * M * rows);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * k);
        rms_norm_apply(input + m * k, gamma, xn, k);
        float* proj = (float*)malloc(sizeof(float) * n_total);
        gemv_ref_fp16(xn, qkv_weightFP16, proj, n_total, k);
        qkv_rope_ref(proj, qkv_theta, qref + m * heads * dim, kref + m * rows, vref + m * rows,
                     n_total, k_offset, v_offset, dim, m);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * heads * dim, sizeof(float));
    uint16_t* kCache = (uint16_t*)calloc(M * rows, sizeof(uint16_t));
    uint16_t* vCache = (uint16_t*)calloc(M * rows, sizeof(uint16_t));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * k * n_total);
    transpose_block16((uint8_t*)qkv_weightFP16, (uint8_t*)transposed, k, n_total, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * k * n_total, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (dim / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint16_t) * M * rows, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint16_t) * M * rows, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 7);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-GEMM-FP16.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer}, .bufferCount = 7,
         .pushConstants = {M, n_total, k, 0, k_offset, v_offset, M}, .pushConstantCount = 7,
         .dispatchX = heads + 2 * kv_heads, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    float* kstored = (float*)malloc(sizeof(float) * M * rows);
    float* vstored = (float*)malloc(sizeof(float) * M * rows);
    for (int i = 0; i < M * rows; i++) kstored[i] = fp16_to_float(kCache[i]);
    for (int i = 0; i < M * rows; i++) vstored[i] = fp16_to_float(vCache[i]);

    report("QKV-Rope-GEMM FP16", 100, qOut, qref, M * heads * dim, ms);
    report("QKV-Rope-GEMM-kv FP16", 100, kstored, kref, M * rows, ms);
    report("QKV-Rope-GEMM-v FP16", 100, vstored, vref, M * rows, ms);

    destroy_buffers(s, bufs, 7);
    free(qref); free(kref); free(vref);
    free(qOut); free(kCache); free(vCache);
    free(kstored); free(vstored);
}

void validateQkvRopeGEMMINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta, int M) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int n_total = (heads + 2 * kv_heads) * dim;
    int k_offset = heads * dim;
    int v_offset = (heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int scaleCount = k * n_total / qkv_weightINT8.group_size;

    float* qref = (float*)malloc(sizeof(float) * M * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * M * rows);
    float* vref = (float*)malloc(sizeof(float) * M * rows);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * k);
        rms_norm_apply(input + m * k, gamma, xn, k);
        float* proj = (float*)malloc(sizeof(float) * n_total);
        gemv_ref_int8(xn, &qkv_weightINT8, proj, n_total, k);
        qkv_rope_ref(proj, qkv_theta, qref + m * heads * dim, kref + m * rows, vref + m * rows,
                     n_total, k_offset, v_offset, dim, m);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * heads * dim, sizeof(float));
    uint16_t* kCache = (uint16_t*)calloc(M * rows, sizeof(uint16_t));
    uint16_t* vCache = (uint16_t*)calloc(M * rows, sizeof(uint16_t));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total);
    transpose_block16(qkv_weightINT8.data, transposed, k, n_total, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (dim / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint16_t) * M * rows, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint16_t) * M * rows, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-GEMM-INT8.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer}, .bufferCount = 9,
         .pushConstants = {M, n_total, k, 0, k_offset, v_offset, M}, .pushConstantCount = 7,
         .dispatchX = heads + 2 * kv_heads, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    float* kstored = (float*)malloc(sizeof(float) * M * rows);
    float* vstored = (float*)malloc(sizeof(float) * M * rows);
    for (int i = 0; i < M * rows; i++) kstored[i] = fp16_to_float(kCache[i]);
    for (int i = 0; i < M * rows; i++) vstored[i] = fp16_to_float(vCache[i]);

    report("QKV-Rope-GEMM INT8", 100, qOut, qref, M * heads * dim, ms);
    report("QKV-Rope-GEMM-kv INT8", 100, kstored, kref, M * rows, ms);
    report("QKV-Rope-GEMM-v INT8", 100, vstored, vref, M * rows, ms);

    destroy_buffers(s, bufs, 9);
    free(qref); free(kref); free(vref);
    free(qOut); free(kCache); free(vCache);
    free(kstored); free(vstored);
}

void validateQkvRopeGEMMINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta, int M) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int n_total = (heads + 2 * kv_heads) * dim;
    int k_offset = heads * dim;
    int v_offset = (heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int scaleCount = k * n_total / qkv_weightINT4.group_size;

    float* qref = (float*)malloc(sizeof(float) * M * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * M * rows);
    float* vref = (float*)malloc(sizeof(float) * M * rows);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * k);
        rms_norm_apply(input + m * k, gamma, xn, k);
        float* proj = (float*)malloc(sizeof(float) * n_total);
        gemv_ref_int4(xn, &qkv_weightINT4, proj, n_total, k);
        qkv_rope_ref(proj, qkv_theta, qref + m * heads * dim, kref + m * rows, vref + m * rows,
                     n_total, k_offset, v_offset, dim, m);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc(M * rows, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(M * rows, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * M, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * M, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * M, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * M, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total / 2);
    transpose_block16(qkv_weightINT4.data, transposed, k, n_total, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total / 2, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (dim / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * M * rows, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * M * rows, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 13);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-GEMM-INT4.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer}, .bufferCount = 13,
         .pushConstants = {M, n_total, k, 0, k_offset, v_offset, M}, .pushConstantCount = 7,
         .dispatchX = heads + 2 * kv_heads, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * M * rows);
    float* vstored = (float*)malloc(sizeof(float) * M * rows);
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < rows; r++) {
            int h = r / dim;
            kstored[m * rows + r] = (float)kCache[m * rows + r] * kScale[h * M + m] - kZero[h * M + m];
            vstored[m * rows + r] = (float)vCache[m * rows + r] * vScale[h * M + m] - vZero[h * M + m];
        }
    }

    report("QKV-Rope-GEMM INT4", 100, qOut, qref, M * heads * dim, ms);
    report("QKV-Rope-GEMM-kv INT4", 100, kstored, kref, M * rows, ms);
    report("QKV-Rope-GEMM-v INT4", 100, vstored, vref, M * rows, ms);

    destroy_buffers(s, bufs, 13);
    free(qref); free(kref); free(vref);
    free(qOut); free(kCache); free(vCache);
    free(kScale); free(kZero); free(vScale); free(vZero);
    free(kstored); free(vstored);
}

static int lmhead_argmax_ref_fp16(const float* x, const uint16_t* w, int n, int k) {
    float best = -FLT_MAX;
    int bestIdx = 0;
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += x[i] * fp16_to_float(w[i * n + j]);
        if (acc > best) {
            best = acc;
            bestIdx = j;
        }
    }
    return bestIdx;
}

void validateLmHeadArgMaxFP16(session s, int vocabSize, int K, float* input, uint16_t* lmHeadFP16) {
    int numGroups = (vocabSize + 255) / 256;
    float* maxValues = (float*)calloc(numGroups, sizeof(float));
    uint32_t* maxIndices = (uint32_t*)calloc(numGroups, sizeof(uint32_t));
    uint32_t* result = (uint32_t*)calloc(1, sizeof(uint32_t));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * vocabSize);
    transpose_block16((uint8_t*)lmHeadFP16, (uint8_t*)transposed, K, vocabSize, QUANT_FP16);

    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * vocabSize, MEMORY_RAM);
    buffer maxValueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, maxValues, sizeof(float) * numGroups, MEMORY_VRAM);
    buffer maxIndexBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, maxIndices, sizeof(uint32_t) * numGroups, MEMORY_VRAM);
    buffer resultBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, result, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {inputBuffer, weightBuffer, maxValueBuffer, maxIndexBuffer, resultBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "LMHead-GEMV-ArgMax-FP16.spv", .buffers = {inputBuffer, weightBuffer, maxValueBuffer, maxIndexBuffer}, .bufferCount = 4,
         .pushConstants = {1, vocabSize, K}, .pushConstantCount = 3,
         .dispatchX = numGroups, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "ArgMax-Reduce.spv", .buffers = {maxValueBuffer, maxIndexBuffer, resultBuffer}, .bufferCount = 3,
         .pushConstants = {vocabSize}, .pushConstantCount = 1,
         .dispatchX = 1, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    int ref = lmhead_argmax_ref_fp16(input, lmHeadFP16, vocabSize, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, resultBuffer, result);
    printf("LMHead-ArgMax FP16: shader[0]= %u ref= %d\n", result[0], ref);
    printf("LMHead-ArgMax FP16 time: %.3f ms\n", ms);

    destroy_buffers(s, bufs, 5);
    free(maxValues);
    free(maxIndices);
    free(result);
}
