//
// Created by deeroot on 11/20/26.
//

#include "../include/model.h"
#include <stdlib.h>
#include "../include/helpers.h"
#include <math.h>

//randomly initialize the embeddings to floats within some small range centered around 0
int model_init(struct transformer *t, size_t vocab_size, size_t dim,
    size_t max_context, size_t num_layers, size_t num_heads) {

    if (t == NULL || dim == 0 || vocab_size == 0) {
        return -1;
    }

    t->vocab_size = vocab_size;
    t->dim = dim;
    t->max_context = max_context;
    t->num_layers = num_layers;
    t->num_heads= num_heads;

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
    size_t d = t->dim;
    float *x = malloc(seq_len * d * sizeof(float));   // residual stream, seq_len × d

    for (size_t i = 0; i < seq_len; i++)
        for (size_t j = 0; j < d; j++)
            //grabbing the values from meaning+position embeddings for each input token
                x[i*d + j] = t->token_embeddings[token_ids[i]*d + j] + t->position_embeddings[i*d + j];
    return x;
}

static void attention_forward(const struct transformer_block *b, float *input_mat,
    size_t seq_len, size_t dim, size_t num_heads) {

    float *Q = malloc(seq_len * dim * sizeof(float)); // Queries: Ex: Q[2] = "sat" asking: who did the sitting?
    float *K = malloc(seq_len * dim * sizeof(float)); // Keys: The answers to the queries, kinda
    float *V = malloc(seq_len * dim * sizeof(float)); // Values: One value per token. Values are Multiplied against
                                                          // weights to obtain attn_out matrix (returned info)

    float *head_outputs = malloc(seq_len * dim * sizeof(float)); // Foreach token, the blend of earlier tokens V rows
    float *attn_update = malloc(seq_len * dim * sizeof(float)); // Combination of attn_out from each head. these
    // referred to as deltaE(sub i) in 3Blue1Brown vids             // are the values which get added to input matrix

    float *scores = malloc(seq_len * sizeof(float)); // Ex: scores = [Q[2]*K[0], Q[2]*K[1], Q[2]*K[2]] = [2.0, ...]
    float *weights = malloc(seq_len * sizeof(float)); // Ex: weights = softmax(scores) = [0.11, ...]

    // getting the learned weights into each matrix as starting values
    matmul(input_mat, b->query_weights, Q, seq_len, dim, dim);
    matmul(input_mat, b->key_weights, K, seq_len,dim, dim);
    matmul(input_mat, b->value_weights, V, seq_len,dim, dim);

    size_t head_dim = dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim); // Apparently dividing by sqrt(query dimension space) helps the math

    for (size_t head = 0; head < num_heads; head++) {
        size_t head_offset = head * head_dim;

        for (size_t q_idx = 0; q_idx<seq_len; q_idx++) {
            // score_token against itself and earlier (Doesn't "cheat" and look ahead at future tokens)
            for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                float dot = 0.0f;
                for (size_t col = 0; col < head_dim; col++) {
                    dot += Q[q_idx*dim + head_offset + col]*K[k_idx*dim + head_offset + col];
                }
                scores[k_idx] = dot * scale;
            }
            softmax(scores, weights, q_idx+1);

            for (size_t col = 0; col < head_dim; col++) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                    // sum of value vector values scaled by the weights
                    sum += weights[k_idx] * V[k_idx*dim + head_offset + col];
                }
                head_outputs[q_idx*dim + head_offset + col] = sum;
            }
        }
    }
    matmul(head_outputs, b->attn_output_weights, attn_update, seq_len, dim, dim);
    for (size_t i = 0; i < seq_len*dim; i++) {
        input_mat[i] += attn_update[i]; //updating original input embeddings.
    }

    free(Q); free(K); free(V); free(head_outputs); free(attn_update);
    free(scores); free(weights);
}



float *model_forward(struct transformer *t, float *input_mat, size_t seq_len) {

    size_t dim = t->dim;

    for (size_t layer = 0; layer < t->num_layers; layer++) {
        attention_forward(&t->blocks[layer], input_mat, seq_len, dim, t->num_heads);
        mlp_forward(&t->blocks[layer], input_mat, seq_len, dim);
    }

    //unembedding: turning each vector back into one score per word
    float *logits = malloc(seq_len * t->vocab_size * sizeof(float));
    matmul(input_mat, t->unembedding_weights, logits, seq_len, dim, t->vocab_size);

    return logits;
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