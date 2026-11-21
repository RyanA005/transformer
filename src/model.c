//
// Created by deeroot on 11/20/26.
//

#include "../include/model.h"
#include <stdlib.h>
#include "../include/helpers.h"

//randomly initialize the embeddings to floats within some small range centered around 0
int model_init(struct transformer *t, size_t vocab_size, size_t model_dim) {
    if (t == NULL || model_dim == 0 || vocab_size == 0) {
        return -1;
    }

    t->vocab_size = vocab_size;
    t->model_dim = model_dim;

    size_t total_elements = vocab_size * model_dim;
    t->embedding_matrix = malloc((total_elements) * sizeof(float));
    if (t->embedding_matrix == NULL) {
        return -1;
    }

    for (size_t i = 0; i < total_elements; i++) {
        t->embedding_matrix[i] = rand_uniform(0.02f);
    }

    return 0;
}

void model_free(struct transformer *t) {
    free(t->embedding_matrix);
    t->embedding_matrix = NULL;
}