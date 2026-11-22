//
// Created by deeroot on 11/22/26.
//

#ifndef TRANSFORMER_NORMALIZATION_H
#define TRANSFORMER_NORMALIZATION_H

#include <stddef.h>


void layer_norm(const float * input, float *output, const float *gamma, const float *beta, size_t seq_len, size_t dim);




#endif //TRANSFORMER_NORMALIZATION_H
