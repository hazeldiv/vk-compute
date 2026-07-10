#include <stdlib.h>
#include <stdint.h>
#include <string.h>

float get_pseudo_random(int i, int j, int seed) {
    unsigned int hash = (unsigned int)(i * 73129 + j * 95121 + seed * 15473);
    hash = (hash ^ 61) ^ (hash >> 16);
    hash += (hash << 3);
    hash ^= (hash >> 4);
    hash *= 0x27d4eb2d;
    hash ^= (hash >> 15);
    return ((float)(hash % 2000001) / 1000000.0f) - 1.0f;
}

float* getData(int seed, int M, int N) {
    float* A = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = get_pseudo_random(i, j, seed);
        }
    }

    return A;
}

uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= -15) {
        if (exp < -24) {
            return sign;
        }
        mant = mant | 0x800000;
        return sign | (mant >> (-14 - exp));
    }
    
    if (exp >= 16) {
        return sign | 0x7C00;
    }
    
    return sign | ((exp + 15) << 10) | (mant >> 13);
}

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x001F;
    uint32_t mant = h & 0x03FF;
    
    uint32_t res_bits;

    if (exp == 0) {
        if (mant == 0) {
            res_bits = sign;
        } else {
            while (!(mant & 0x0400)) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x0400;
            res_bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        res_bits = sign | (0xFF << 23) | (mant << 13);
    } else {
        // Normal number
        res_bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &res_bits, sizeof(f));
    return f;
}

uint16_t* getDataFP16(int seed, int M, int N) {
    uint16_t* A = (uint16_t*)malloc(sizeof(uint16_t) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = get_pseudo_random(i, j, seed);
            A[i * N + j] = float_to_fp16(val);
        }
    }
    return A;
}

void transpose(const float* src, float* dest, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int src_idx = i * n + j;
            int dest_idx = j * m + i;
            
            dest[dest_idx] = src[src_idx];
        }
    }
}

void transpose_block4(const float *input, float *output, int M, int N) {
    int out_N = N * 4;
    for (int b = 0; b < M; b += 4) {
        int out_row = b / 4;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < 4; ++k) {
                int in_idx = (b + k) * N + j;
                int out_idx = out_row * out_N + (j * 4) + k;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

void transpose_block4_FP16(const uint16_t *input, uint16_t *output, int M, int N) {
    int out_N = N * 8;
    for (int b = 0; b < M; b += 8) {
        int out_row = b / 8;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < 8; ++k) {
                int in_idx = (b + k) * N + j;
                int out_idx = out_row * out_N + (j * 8) + k;
                output[out_idx] = input[in_idx];
            }
        }
    }
}