#ifndef HELPERS_H
#define HELPERS_H

#include <stdint.h>

void matmul(float *a, float *b, float *c, uint64_t M, uint64_t K, uint64_t N);
void matprint(float *a, uint64_t M, uint64_t N);
void softmax(float *z, float *p, uint64_t N);

#endif
