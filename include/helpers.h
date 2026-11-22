#ifndef HELPERS_H
#define HELPERS_H

#include <stdint.h>
#include <stddef.h>


void matmul(float *a, float *b, float *c, uint64_t M, uint64_t K, uint64_t N);
void matprint(float *a, uint64_t M, uint64_t N);
void softmax(float *z, float *p, uint64_t N);
float fast_gelu(float x);
float rand01(void);
float rand_uniform(float scale);
float *alloc_random_matrix(size_t rows, size_t cols);
float *alloc_filled_vector(size_t n, float value);


#endif
