
#ifndef MODEL_H
#define MODEL_H
#include <stddef.h>



struct transformer {
    size_t vocab_size, model_dim;
    float *embedding_matrix;
    //more matrices later: position, key weights, query weights, ...there will be a bunch
    //attention blocks, MLP layers
};

//randomly initialize the embeddings to floats within some small range centered around 0
int model_init(struct transformer *t, size_t vocab_size, size_t model_dim);

float *model_forward(struct transformer *t, const float *input_embeds, size_t n);

int model_save(const struct transformer *t, const char *path);

void model_free(struct transformer *t);

#endif //MODEL_H