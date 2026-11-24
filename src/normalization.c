//
// Created by deeroot on 11/22/26.
//
#include "../include/normalization.h"
#include <math.h>

void layer_norm(const float * input, float *output, const float *gamma, const float *beta, size_t seq_len, size_t dim) {
    /*
        Rescales each tokens vector to mean 0 and variance of 1, then applies a learned scale and shift.

        gamma: scale. This will stretch or shrink each column (values). One per column
        beta: shift. This moves columns up or down (the values). One per column
        So *gamma and *beta are both size dim.


        After normalizing each dimension/feature, these two variables are shifting/scaling,
        one token (separately) at a time.

        The same gamma and beta are reused for every token (row).
     */
    const float variance_epsilon = 1e-5f;

    for (size_t row = 0; row < seq_len; row++) {
        const float *in_row = input + row * dim;
        float *out_row = output + row * dim;

        float mean = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            mean += in_row[i];
        }
        mean /= dim;

        float variance = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            variance += (in_row[i] - mean) * (in_row[i] - mean);
        }
        variance /= dim;

        float inverse_std_dev = 1.0f / sqrtf(variance + variance_epsilon);

        for (size_t i = 0; i < dim; i++) {
            out_row[i] = (in_row[i] - mean) * inverse_std_dev * gamma[i] + beta[i];
        }
    }
}
