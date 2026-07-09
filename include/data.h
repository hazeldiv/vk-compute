#ifndef data_h
#define data_h
#include <stdlib.h>
#include <stdint.h>

float* getData(int seed, int M, int N);
uint16_t* getDataFP16(int seed, int M, int N);
void transpose(const float* src, float* dest, int m, int n);
void transpose_block4(const float *input, float *output, int M, int N);
void transpose_block4_FP16(const uint16_t *input, uint16_t *output, int M, int N);

#endif