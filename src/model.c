//
// Created by deeroot on 11/20/26.
//

#include "../include/model.h"
#include <stdlib.h>
#include "../include/helpers.h"

//randomly initialize the embeddings to floats within some small range centered around 0
int model_init(struct transformer *t, size_t vocab_size, size_t dim, size_t max_context, size_t num_layers) {

    if (t == NULL || dim == 0 || vocab_size == 0) {
        return -1;
    }

    t->vocab_size = vocab_size;
    t->model_dim = dim;
    t->max_context = max_context;
    t->num_layers = num_layers;

    t->token_embeddings    = alloc_random_matrix(vocab_size, dim);
    t->position_embeddings = alloc_random_matrix(max_context, dim);
    t->unembedding_weights = alloc_random_matrix(dim, vocab_size);

    t->blocks = malloc(num_layers * sizeof(struct transformer_block));
    for (size_t l = 0; l < num_layers; l++) {
        struct transformer_block *b = &t->blocks[l];
        b->query_weights       = alloc_random_matrix(dim, dim);
        b->key_weights         = alloc_random_matrix(dim, dim);
        b->value_weights       = alloc_random_matrix(dim, dim);
        b->attn_output_weights = alloc_random_matrix(dim, dim);
        b->mlp_up_weights      = alloc_random_matrix(dim, 4 * dim);
        b->mlp_down_weights    = alloc_random_matrix(4 * dim, dim);
    }

    return 0;
}

float *get_input_embeddings(const struct transformer *t, const int *token_ids, size_t seq_len) {
    /*
     - token_ids: The input to the forward pass. Ex: {6, 3} -> predicts 5.
                    In llm it would then model_forward(t, {6, 3, 5}, 3)
     - seq_len: Number of input tokens

     This is a helper method that really could be inside of model_forward

     Takes token ids and returns matrix of the embedding float values (meaning+position)
     */
    size_t d = t->model_dim;
    float *x = malloc(seq_len * d * sizeof(float));   // residual stream, seq_len × d

    for (size_t i = 0; i < seq_len; i++)
        for (size_t j = 0; j < d; j++)
            //grabbing the values from meaning+position embeddings for each input token
                x[i*d + j] = t->token_embeddings[token_ids[i]*d + j] + t->position_embeddings[i*d + j];
    return x;
}

float *model_forward(struct transformer *t, float *input_embeds, size_t seq_len) {
    (void)t;
    (void)seq_len;
    // TODO: blocks, then unembedding
    return input_embeds;
}

void model_free(struct transformer *t) {
    for (size_t l = 0; l < t->num_layers; l++) {
        struct transformer_block *b = &t->blocks[l];
        free(b->query_weights);
        free(b->key_weights);
        free(b->value_weights);
        free(b->attn_output_weights);
        free(b->mlp_up_weights);
        free(b->mlp_down_weights);
    }
    free(t->blocks);
    free(t->token_embeddings);
    free(t->position_embeddings);
    free(t->unembedding_weights);
}