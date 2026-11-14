#include <math.h>
#include <stdio.h>

#include "helpers.h"

void matmul(float *a, float *b, float *c, uint64_t M, uint64_t K, uint64_t N) {
/*
 * a -> M x K
 * b -> K x N
 * c -> M x N
 *
 */
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++)
            c[i * N + j] = 0.0f;
        for (uint64_t k = 0; k < K; k++) {
            float t = a[i * K + k];
            for (uint64_t j = 0; j < N; j++)
                c[i * N + j] += t * b[k * N + j];
        }
    }
}

void matprint(float *a, uint64_t M, uint64_t N) {
    /*
     * a -> M x N
     */
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++)
            printf("%.3f ", a[i * N + j]);
        printf("\n");
    }
}

void softmax(float *z, float *p, uint64_t N) {
    /*
     * takes input buffer z, copies probabilities into initialized p buffer
     */
    float max = z[0];
    for (uint64_t i = 1; i < N; i++)
        if (z[i] > max)
            max = z[i];

    float sum = 0.0f;
    for (uint64_t i = 0; i < N; i++) {
        p[i] = expf(z[i] - max);
        sum += p[i];
    }
    for (uint64_t i = 0; i < N; i++)
        p[i] /= sum;
}

float fast_gelu(float x) {
    /*
     * fast and advantaged over RELU
     */
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2 / M_PI)
    const float coeff = 0.044715f;
    
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}
