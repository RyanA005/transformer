//
// Created by deeroot on 11/20/26.
//

#include "../include/model.h"
#include <stdlib.h>
#include <string.h>
#include "../include/helpers.h"
#include "../include/normalization.h"
#include <math.h>

static float *alloc_buf(size_t n_floats) {
    return malloc(n_floats * sizeof(float));
}

static float **alloc_layer_bufs(size_t num_layers, size_t n_floats) {
    float **arr = calloc(num_layers, sizeof(float *));
    if (arr == NULL) return NULL;
    for (size_t l = 0; l < num_layers; l++) {
        arr[l] = alloc_buf(n_floats);
        if (arr[l] == NULL) {
            for (size_t i = 0; i < l; i++) free(arr[i]);
            free(arr);
            return NULL;
        }
    }
    return arr;
}

static void free_layer_bufs(float **arr, size_t num_layers) {
    if (arr == NULL) return;
    for (size_t l = 0; l < num_layers; l++) free(arr[l]);
    free(arr);
}

// randomly initialize the embeddings to floats within some small range centered around 0
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

void runtime_free(struct runtime *rt) {
    if (rt == NULL) return;

    free_layer_bufs(rt->layer_residuals, rt->num_layers);
    free_layer_bufs(rt->attn_in, rt->num_layers);
    free_layer_bufs(rt->attn_normed, rt->num_layers);
    free_layer_bufs(rt->Q, rt->num_layers);
    free_layer_bufs(rt->K, rt->num_layers);
    free_layer_bufs(rt->V, rt->num_layers);
    free_layer_bufs(rt->head_outputs, rt->num_layers);
    free_layer_bufs(rt->attn_update, rt->num_layers);
    free_layer_bufs(rt->attn_probs, rt->num_layers);
    free_layer_bufs(rt->mlp_in, rt->num_layers);
    free_layer_bufs(rt->mlp_normed, rt->num_layers);
    free_layer_bufs(rt->mlp_pre_gelu, rt->num_layers);
    free_layer_bufs(rt->mlp_hidden, rt->num_layers);
    free_layer_bufs(rt->mlp_post_update, rt->num_layers);

    free(rt->residual);
    free(rt->final_normed);
    free(rt->logits);
    free(rt->probs);
    free(rt->dlogits);
    free(rt->scores);
    free(rt->attn_weights);
    free(rt->dresidual);
    free(rt->dscratch);
    free(rt->dscratch2);
    free(rt->dhidden);
    memset(rt, 0, sizeof(*rt));
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
    rt->num_heads = t->num_heads;
    rt->seq_len = 0;

    size_t max_seq = rt->max_seq_len;
    size_t dim = rt->dim;
    size_t vocab = rt->vocab_size;
    size_t layers = rt->num_layers;
    size_t heads = rt->num_heads;
    size_t hidden_dim = 4 * dim;
    size_t resid_n = max_seq * dim;
    // one causal seq×seq block per head
    size_t attn_n = heads * max_seq * max_seq;
    size_t hidden_n = max_seq * hidden_dim;

    rt->residual = alloc_buf(resid_n);
    rt->final_normed = alloc_buf(resid_n);
    rt->logits = alloc_buf(max_seq * vocab);
    rt->probs = alloc_buf(vocab);
    rt->dlogits = alloc_buf(max_seq * vocab);
    rt->scores = alloc_buf(max_seq);
    rt->attn_weights = alloc_buf(max_seq);
    rt->dresidual = alloc_buf(resid_n);
    rt->dscratch = alloc_buf(resid_n);
    rt->dscratch2 = alloc_buf(resid_n);
    rt->dhidden = alloc_buf(hidden_n);

    rt->layer_residuals = alloc_layer_bufs(layers, resid_n);
    rt->attn_in = alloc_layer_bufs(layers, resid_n);
    rt->attn_normed = alloc_layer_bufs(layers, resid_n);
    rt->Q = alloc_layer_bufs(layers, resid_n);
    rt->K = alloc_layer_bufs(layers, resid_n);
    rt->V = alloc_layer_bufs(layers, resid_n);
    rt->head_outputs = alloc_layer_bufs(layers, resid_n);
    rt->attn_update = alloc_layer_bufs(layers, resid_n);
    rt->attn_probs = alloc_layer_bufs(layers, attn_n);
    rt->mlp_in = alloc_layer_bufs(layers, resid_n);
    rt->mlp_normed = alloc_layer_bufs(layers, resid_n);
    rt->mlp_pre_gelu = alloc_layer_bufs(layers, hidden_n);
    rt->mlp_hidden = alloc_layer_bufs(layers, hidden_n);
    rt->mlp_post_update = alloc_layer_bufs(layers, resid_n);

    if (rt->residual == NULL || rt->final_normed == NULL ||
        rt->logits == NULL || rt->probs == NULL || rt->dlogits == NULL ||
        rt->scores == NULL || rt->attn_weights == NULL ||
        rt->dresidual == NULL || rt->dscratch == NULL || rt->dscratch2 == NULL ||
        rt->dhidden == NULL ||
        rt->layer_residuals == NULL || rt->attn_in == NULL || rt->attn_normed == NULL ||
        rt->Q == NULL || rt->K == NULL || rt->V == NULL ||
        rt->head_outputs == NULL || rt->attn_update == NULL || rt->attn_probs == NULL ||
        rt->mlp_in == NULL || rt->mlp_normed == NULL ||
        rt->mlp_pre_gelu == NULL || rt->mlp_hidden == NULL ||
        rt->mlp_post_update == NULL) {
        runtime_free(rt);
        return -1;
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
                              size_t layer, size_t num_heads) {
    size_t seq_len = rt->seq_len;
    size_t dim = rt->dim;
    size_t resid_bytes = seq_len * dim * sizeof(float);

    // snapshot residual before we touch it — backward needs the LN input
    memcpy(rt->attn_in[layer], rt->residual, resid_bytes);

    layer_norm(rt->attn_in[layer], rt->attn_normed[layer],
               b->attn_norm_gamma, b->attn_norm_beta, seq_len, dim);

    matmul(rt->attn_normed[layer], b->query_weights, rt->Q[layer], seq_len, dim, dim);
    matmul(rt->attn_normed[layer], b->key_weights, rt->K[layer], seq_len, dim, dim);
    matmul(rt->attn_normed[layer], b->value_weights, rt->V[layer], seq_len, dim, dim);

    size_t head_dim = dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // zero attn_probs so unused upper triangle stays clean for debugging
    memset(rt->attn_probs[layer], 0,
           num_heads * rt->max_seq_len * rt->max_seq_len * sizeof(float));

    for (size_t head = 0; head < num_heads; head++) {
        size_t head_offset = head * head_dim;

        for (size_t q_idx = 0; q_idx < seq_len; q_idx++) {
            // causal: only score against itself and earlier tokens (no looking ahead)
            for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                float dot = 0.0f;
                for (size_t col = 0; col < head_dim; col++) {
                    dot += rt->Q[layer][q_idx * dim + head_offset + col] *
                           rt->K[layer][k_idx * dim + head_offset + col];
                }
                rt->scores[k_idx] = dot * scale;
            }
            softmax(rt->scores, rt->attn_weights, q_idx + 1);

            // stash this head's probs — each head looks at different Q/K slices
            float *prob_row = rt->attn_probs[layer] +
                (head * rt->max_seq_len + q_idx) * rt->max_seq_len;
            for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                prob_row[k_idx] = rt->attn_weights[k_idx];
            }

            for (size_t col = 0; col < head_dim; col++) {
                float sum = 0.0f;
                for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                    sum += rt->attn_weights[k_idx] *
                           rt->V[layer][k_idx * dim + head_offset + col];
                }
                rt->head_outputs[layer][q_idx * dim + head_offset + col] = sum;
            }
        }
    }

    matmul(rt->head_outputs[layer], b->attn_output_weights, rt->attn_update[layer],
           seq_len, dim, dim);
    for (size_t i = 0; i < seq_len * dim; i++) {
        rt->residual[i] += rt->attn_update[layer][i];
    }
}

static void mlp_forward(const struct transformer_block *b, struct runtime *rt,
                        size_t layer) {
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
    size_t resid_bytes = seq_len * dim * sizeof(float);

    memcpy(rt->mlp_in[layer], rt->residual, resid_bytes);

    layer_norm(rt->mlp_in[layer], rt->mlp_normed[layer],
               b->mlp_norm_gamma, b->mlp_norm_beta, seq_len, dim);

    matmul(rt->mlp_normed[layer], b->mlp_up_weights, rt->mlp_pre_gelu[layer],
           seq_len, dim, hidden_dim);

    // keep pre-GELU for backward; mlp_hidden is the activated copy
    for (size_t i = 0; i < seq_len * hidden_dim; i++) {
        rt->mlp_hidden[layer][i] = fast_gelu(rt->mlp_pre_gelu[layer][i]);
    }

    matmul(rt->mlp_hidden[layer], b->mlp_down_weights, rt->mlp_post_update[layer],
           seq_len, hidden_dim, dim);

    for (size_t i = 0; i < seq_len * dim; i++) {
        rt->residual[i] += rt->mlp_post_update[layer][i];
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
        attention_forward(&t->blocks[layer], rt, layer, t->num_heads);
        mlp_forward(&t->blocks[layer], rt, layer);
        memcpy(rt->layer_residuals[layer], rt->residual,
               seq_len * t->dim * sizeof(float));
    }

    layer_norm(rt->residual, rt->final_normed, t->final_norm_gamma, t->final_norm_beta,
               seq_len, t->dim);
    matmul(rt->final_normed, t->unembedding_weights, rt->logits,
           seq_len, t->dim, t->vocab_size);

    // inference convenience: probs for the last position only
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
