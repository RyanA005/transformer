//
// Created by deeroot on 11/20/26.
//

#include "../include/model.h"
#include <stdlib.h>
#include <string.h>
#include "../include/helpers.h"
#include "../include/normalization.h"
#include <math.h>

//randomly initialize the embeddings to floats within some small range centered around 4
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

        b->attn_norm_gamma = alloc_filled_vector(dim, 1.0f);
        b->attn_norm_beta  = alloc_filled_vector(dim, 0.0f);
        b->mlp_norm_gamma  = alloc_filled_vector(dim, 1.0f);
        b->mlp_norm_beta   = alloc_filled_vector(dim, 0.0f);
    }

    t->final_norm_gamma = alloc_filled_vector(dim, 1.0f);
    t->final_norm_beta  = alloc_filled_vector(dim, 0.0f);

    return 0;
}

int runtime_init(struct runtime *rt, const struct transformer *t) {
    if (rt == NULL || t == NULL || t->max_context == 0 || t->dim == 0) {
        return -1;
    }

    memset(rt, 0, sizeof(*rt));
    rt->max_seq_len = t->max_context;
    rt->dim = t->dim;
    rt->vocab_size = t->vocab_size;
    rt->num_layers = t->num_layers;
    rt->seq_len = 0;

    size_t max_seq = rt->max_seq_len;
    size_t dim = rt->dim;
    size_t vocab = rt->vocab_size;
    size_t hidden_dim = 4 * dim;
    size_t resid_bytes = max_seq * dim * sizeof(float);

    rt->residual = malloc(resid_bytes);
    rt->normalized = malloc(resid_bytes);
    rt->Q = malloc(resid_bytes);
    rt->K = malloc(resid_bytes);
    rt->V = malloc(resid_bytes);
    rt->head_outputs = malloc(resid_bytes);
    rt->attn_update = malloc(resid_bytes);
    rt->scores = malloc(max_seq * sizeof(float));
    rt->attn_weights = malloc(max_seq * sizeof(float));
    rt->mlp_hidden = malloc(max_seq * hidden_dim * sizeof(float));
    rt->mlp_update = malloc(resid_bytes);
    rt->logits = malloc(max_seq * vocab * sizeof(float));
    rt->probs = malloc(vocab * sizeof(float));
    rt->layer_residuals = calloc(rt->num_layers, sizeof(float *));

    if (rt->residual == NULL || rt->normalized == NULL ||
        rt->Q == NULL || rt->K == NULL || rt->V == NULL ||
        rt->head_outputs == NULL || rt->attn_update == NULL ||
        rt->scores == NULL || rt->attn_weights == NULL ||
        rt->mlp_hidden == NULL || rt->mlp_update == NULL ||
        rt->logits == NULL || rt->probs == NULL ||
        rt->layer_residuals == NULL) {
        runtime_free(rt);
        return -1;
    }

    for (size_t l = 0; l < rt->num_layers; l++) {
        rt->layer_residuals[l] = malloc(resid_bytes);
        if (rt->layer_residuals[l] == NULL) {
            runtime_free(rt);
            return -1;
        }
    }

    return 0;
}

static void embed_tokens(const struct transformer *t, struct runtime *rt,
                         const int *token_ids, size_t seq_len) {
    size_t d = t->dim;
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d; j++) {
            rt->residual[i * d + j] =
                t->token_embeddings[token_ids[i] * d + j] +
                t->position_embeddings[i * d + j];
        }
    }
}

static void attention_forward(const struct transformer_block *b, struct runtime *rt,
                              size_t num_heads) {
    size_t seq_len = rt->seq_len;
    size_t dim = rt->dim;

    layer_norm(rt->residual, rt->normalized, b->attn_norm_gamma, b->attn_norm_beta,
               seq_len, dim);

    matmul(rt->normalized, b->query_weights, rt->Q, seq_len, dim, dim);
    matmul(rt->normalized, b->key_weights, rt->K, seq_len, dim, dim);
    matmul(rt->normalized, b->value_weights, rt->V, seq_len, dim, dim);

    size_t head_dim = dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (size_t head = 0; head < num_heads; head++) {
        size_t head_offset = head * head_dim;

        for (size_t q_idx = 0; q_idx < seq_len; q_idx++) {
            for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                float dot = 0.0f;
                for (size_t col = 0; col < head_dim; col++) {
                    dot += rt->Q[q_idx * dim + head_offset + col] *
                           rt->K[k_idx * dim + head_offset + col];
                }
                rt->scores[k_idx] = dot * scale;
            }
            softmax(rt->scores, rt->attn_weights, q_idx + 1);

            for (size_t col = 0; col < head_dim; col++) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                    sum += rt->attn_weights[k_idx] *
                           rt->V[k_idx * dim + head_offset + col];
                }
                rt->head_outputs[q_idx * dim + head_offset + col] = sum;
            }
        }
    }

    matmul(rt->head_outputs, b->attn_output_weights, rt->attn_update,
           seq_len, dim, dim);
    for (size_t i = 0; i < seq_len * dim; i++) {
        rt->residual[i] += rt->attn_update[i];
    }
}

static void mlp_forward(const struct transformer_block *b, struct runtime *rt) {
    /*
     Why the projections? The up projection widens each token to 4·dim hidden units so the GELU nonlinearity
     has many neurons to work with, which is where the MLP gets its capacity.
     The down projection combines those units back to dim, because the output has to match the
     residual stream's width to be added back in.

     This MLP, following the OG GPT-3, is two linear layers with a nonlinear function/layer sandwiched in between
     */

    size_t seq_len = rt->seq_len;
    size_t dim = rt->dim;
    size_t hidden_dim = 4 * dim;

    layer_norm(rt->residual, rt->normalized, b->mlp_norm_gamma, b->mlp_norm_beta,
               seq_len, dim);

    matmul(rt->normalized, b->mlp_up_weights, rt->mlp_hidden,
           seq_len, dim, hidden_dim);

    for (size_t i = 0; i < seq_len * hidden_dim; i++) {
        rt->mlp_hidden[i] = fast_gelu(rt->mlp_hidden[i]);
    }

    matmul(rt->mlp_hidden, b->mlp_down_weights, rt->mlp_update,
           seq_len, hidden_dim, dim);

    for (size_t i = 0; i < seq_len * dim; i++) {
        rt->residual[i] += rt->mlp_update[i];
    }
}

int model_forward(const struct transformer *t, struct runtime *rt,
                  const int *token_ids, size_t seq_len) {
    if (t == NULL || rt == NULL || token_ids == NULL ||
        seq_len == 0 || seq_len > rt->max_seq_len ||
        rt->dim != t->dim || rt->vocab_size != t->vocab_size ||
        rt->num_layers != t->num_layers || rt->max_seq_len != t->max_context) {
        return -1;
    }

    rt->seq_len = seq_len;
    embed_tokens(t, rt, token_ids, seq_len);

    for (size_t layer = 0; layer < t->num_layers; layer++) {
        attention_forward(&t->blocks[layer], rt, t->num_heads);
        mlp_forward(&t->blocks[layer], rt);
        memcpy(rt->layer_residuals[layer], rt->residual,
               seq_len * t->dim * sizeof(float));
    }

    layer_norm(rt->residual, rt->normalized, t->final_norm_gamma, t->final_norm_beta,
               seq_len, t->dim);
    matmul(rt->normalized, t->unembedding_weights, rt->logits,
           seq_len, t->dim, t->vocab_size);

    softmax(rt->logits + (seq_len - 1) * t->vocab_size, rt->probs, t->vocab_size);
    return 0;
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
        free(b->attn_norm_gamma); free(b->attn_norm_beta);
        free(b->mlp_norm_gamma);  free(b->mlp_norm_beta);
    }
    free(t->blocks);
    free(t->token_embeddings);
    free(t->position_embeddings);
    free(t->unembedding_weights);
    free(t->final_norm_gamma); free(t->final_norm_beta);
}

void runtime_free(struct runtime *rt) {
    if (rt == NULL) {
        return;
    }
    if (rt->layer_residuals != NULL) {
        for (size_t l = 0; l < rt->num_layers; l++) {
            free(rt->layer_residuals[l]);
        }
        free(rt->layer_residuals);
    }
    free(rt->residual);
    free(rt->normalized);
    free(rt->Q);
    free(rt->K);
    free(rt->V);
    free(rt->head_outputs);
    free(rt->attn_update);
    free(rt->scores);
    free(rt->attn_weights);
    free(rt->mlp_hidden);
    free(rt->mlp_update);
    free(rt->logits);
    free(rt->probs);
    memset(rt, 0, sizeof(*rt));
}
