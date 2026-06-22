#include <stdlib.h>

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