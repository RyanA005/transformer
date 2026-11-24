#include "../include/train.h"
#include "../include/helpers.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float *alloc_zeros(size_t n) {
    float *p = calloc(n, sizeof(float));
    return p;
}

static void zero_n(float *p, size_t n) {
    if (p) memset(p, 0, n * sizeof(float));
}

static void sgd_buf(float *w, const float *dw, size_t n, float lr) {
    for (size_t i = 0; i < n; i++) w[i] -= lr * dw[i];
}

void grads_free(struct grads *g) {
    if (g == NULL) return;
    if (g->blocks != NULL) {
        for (size_t l = 0; l < g->num_layers; l++) {
            struct grads_block *b = &g->blocks[l];
            free(b->query_weights);
            free(b->key_weights);
            free(b->value_weights);
            free(b->attn_output_weights);
            free(b->attn_norm_gamma);
            free(b->attn_norm_beta);
            free(b->mlp_norm_gamma);
            free(b->mlp_norm_beta);
            free(b->mlp_up_weights);
            free(b->mlp_down_weights);
        }
        free(g->blocks);
    }
    free(g->token_embeddings);
    free(g->position_embeddings);
    free(g->final_norm_gamma);
    free(g->final_norm_beta);
    free(g->unembedding_weights);
    memset(g, 0, sizeof(*g));
}

int grads_init(struct grads *g, const struct transformer *t) {
    if (g == NULL || t == NULL) return -1;
    memset(g, 0, sizeof(*g));

    g->vocab_size = t->vocab_size;
    g->dim = t->dim;
    g->max_context = t->max_context;
    g->num_layers = t->num_layers;

    size_t dim = t->dim;
    size_t vocab = t->vocab_size;
    size_t ctx = t->max_context;

    g->token_embeddings = alloc_zeros(vocab * dim);
    g->position_embeddings = alloc_zeros(ctx * dim);
    g->final_norm_gamma = alloc_zeros(dim);
    g->final_norm_beta = alloc_zeros(dim);
    g->unembedding_weights = alloc_zeros(dim * vocab);
    g->blocks = calloc(t->num_layers, sizeof(struct grads_block));

    if (!g->token_embeddings || !g->position_embeddings ||
        !g->final_norm_gamma || !g->final_norm_beta ||
        !g->unembedding_weights || !g->blocks) {
        grads_free(g);
        return -1;
    }

    for (size_t l = 0; l < t->num_layers; l++) {
        struct grads_block *b = &g->blocks[l];
        b->query_weights = alloc_zeros(dim * dim);
        b->key_weights = alloc_zeros(dim * dim);
        b->value_weights = alloc_zeros(dim * dim);
        b->attn_output_weights = alloc_zeros(dim * dim);
        b->attn_norm_gamma = alloc_zeros(dim);
        b->attn_norm_beta = alloc_zeros(dim);
        b->mlp_norm_gamma = alloc_zeros(dim);
        b->mlp_norm_beta = alloc_zeros(dim);
        b->mlp_up_weights = alloc_zeros(dim * 4 * dim);
        b->mlp_down_weights = alloc_zeros(4 * dim * dim);
        if (!b->query_weights || !b->key_weights || !b->value_weights ||
            !b->attn_output_weights || !b->attn_norm_gamma || !b->attn_norm_beta ||
            !b->mlp_norm_gamma || !b->mlp_norm_beta ||
            !b->mlp_up_weights || !b->mlp_down_weights) {
            grads_free(g);
            return -1;
        }
    }
    return 0;
}

void grads_zero(struct grads *g) {
    size_t dim = g->dim;
    size_t vocab = g->vocab_size;
    size_t ctx = g->max_context;

    zero_n(g->token_embeddings, vocab * dim);
    zero_n(g->position_embeddings, ctx * dim);
    zero_n(g->final_norm_gamma, dim);
    zero_n(g->final_norm_beta, dim);
    zero_n(g->unembedding_weights, dim * vocab);

    for (size_t l = 0; l < g->num_layers; l++) {
        struct grads_block *b = &g->blocks[l];
        zero_n(b->query_weights, dim * dim);
        zero_n(b->key_weights, dim * dim);
        zero_n(b->value_weights, dim * dim);
        zero_n(b->attn_output_weights, dim * dim);
        zero_n(b->attn_norm_gamma, dim);
        zero_n(b->attn_norm_beta, dim);
        zero_n(b->mlp_norm_gamma, dim);
        zero_n(b->mlp_norm_beta, dim);
        zero_n(b->mlp_up_weights, dim * 4 * dim);
        zero_n(b->mlp_down_weights, 4 * dim * dim);
    }
}

/*
  C = A @ B
  A: M×K, B: K×N, C: M×N

  Chain rule:
    dA = dC @ B^T     (M×N)(N×K) -> M×K
    dB = A^T @ dC     (K×M)(M×N) -> K×N

  accumulate=1 means += into dA/dB
*/
static void matmul_backward(const float *A, const float *B, const float *dC,
                            float *dA, float *dB,
                            size_t M, size_t K, size_t N,
                            int accumulate_dA, int accumulate_dB) {
    if (dA != NULL) {
        if (!accumulate_dA) zero_n(dA, M * K);
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                float sum = 0.0f;
                for (size_t j = 0; j < N; j++) {
                    sum += dC[i * N + j] * B[k * N + j];
                }
                dA[i * K + k] += sum;
            }
        }
    }
    if (dB != NULL) {
        if (!accumulate_dB) zero_n(dB, K * N);
        for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t i = 0; i < M; i++) {
                    sum += A[i * K + k] * dC[i * N + j];
                }
                dB[k * N + j] += sum;
            }
        }
    }
}

/*
  LayerNorm backward for one sequence of rows.

  forward per row: xhat = (x - mean) / sqrt(var+eps); y = xhat * gamma + beta
  gamma/beta grads accumulate per row
*/
static void layer_norm_backward(const float *x, const float *dout,
                                const float *gamma,
                                float *dx, float *dgamma, float *dbeta,
                                size_t seq_len, size_t dim) {
    /*
      dx is overwritten; dgamma/dbeta accumulate across the step.
    */
    const float eps = 1e-5f;

    for (size_t row = 0; row < seq_len; row++) {
        const float *xr = x + row * dim;
        const float *dor = dout + row * dim;
        float *dxr = dx + row * dim;

        float mean = 0.0f;
        for (size_t i = 0; i < dim; i++) mean += xr[i];
        mean /= (float)dim;

        float var = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float d = xr[i] - mean;
            var += d * d;
        }
        var /= (float)dim;
        float inv_std = 1.0f / sqrtf(var + eps);

        float sum_dxhat = 0.0f;
        float sum_dxhat_xhat = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float xhat = (xr[i] - mean) * inv_std;
            float dxhat = dor[i] * gamma[i];
            if (dgamma) dgamma[i] += dor[i] * xhat;
            if (dbeta) dbeta[i] += dor[i];
            dxr[i] = dxhat; // temp stash
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xhat;
        }

        float mean_dxhat = sum_dxhat / (float)dim;
        float mean_dxhat_xhat = sum_dxhat_xhat / (float)dim;
        for (size_t i = 0; i < dim; i++) {
            float xhat = (xr[i] - mean) * inv_std;
            float dxhat = dxr[i];
            dxr[i] = inv_std * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
        }
    }
}

static float gelu_grad(float x) {
    /*
      derivative of fast GELU
    */
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    float x2 = x * x;
    float x3 = x2 * x;
    float u = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_u = tanhf(u);
    float sech2 = 1.0f - tanh_u * tanh_u;
    float du = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);
    return 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * du;
}

float cross_entropy_next_token(struct runtime *rt, const int *token_ids) {
    /*
      Softmax + CE gradient collapses nice:
        dlogits = (p - one_hot(target)) / n
        if target is 4 and p[4]=0.1, that slot gets (0.1-1)/n = -0.9/n
        every other slot will get p[j]/n - overprediction
    */
    size_t seq_len = rt->seq_len;
    size_t vocab = rt->vocab_size;
    size_t n_targets = seq_len - 1;
    if (n_targets == 0) return -1.0f;

    float loss = 0.0f;
    zero_n(rt->dlogits, seq_len * vocab);

    // last row has no next token target
    for (size_t t = 0; t < n_targets; t++) {
        float *logits_t = rt->logits + t * vocab;
        float *dlogits_t = rt->dlogits + t * vocab;
        // lazy reusing probs buffer as softmax scratch
        softmax(logits_t, rt->probs, vocab);

        int target = token_ids[t + 1];
        // clamp numerical log
        float p_tgt = rt->probs[target];
        if (p_tgt < 1e-12f) p_tgt = 1e-12f;
        loss += -logf(p_tgt);

        for (size_t j = 0; j < vocab; j++) {
            dlogits_t[j] = rt->probs[j] / (float)n_targets;
        }
        dlogits_t[target] -= 1.0f / (float)n_targets;
    }

    // restore last-token probs, remove if we dont need to debug...
    softmax(rt->logits + (seq_len - 1) * vocab, rt->probs, vocab);

    return loss / (float)n_targets;
}

static void mlp_backward(const struct transformer_block *b, struct grads_block *gb,
                         struct runtime *rt, size_t layer) {
    size_t seq_len = rt->seq_len;
    size_t dim = rt->dim;
    size_t hidden_dim = 4 * dim;

    /*
      residual = mlp_in + mlp_post_update
      so d(mlp_post_update) = dresidual, and d(mlp_in) also gets dresidual
    */
    // d_down_out is rt->dresidual flowing into the down-proj output
    // mlp_post = hidden @ W_down
    matmul_backward(rt->mlp_hidden[layer], b->mlp_down_weights, rt->dresidual,
                    rt->dhidden, gb->mlp_down_weights,
                    seq_len, hidden_dim, dim, 0, 1);

    // GELU' elementwise: dpre = dhidden * gelu'(pre)
    for (size_t i = 0; i < seq_len * hidden_dim; i++) {
        rt->dhidden[i] *= gelu_grad(rt->mlp_pre_gelu[layer][i]);
    }

    // pre = normed @ W_up  -> d_normed into dscratch, accumulate W_up grads
    matmul_backward(rt->mlp_normed[layer], b->mlp_up_weights, rt->dhidden,
                    rt->dscratch, gb->mlp_up_weights,
                    seq_len, dim, hidden_dim, 0, 1);

    // LN backward: dout=dscratch, x=mlp_in -> dx into dscratch2
    layer_norm_backward(rt->mlp_in[layer], rt->dscratch,
                        b->mlp_norm_gamma,
                        rt->dscratch2, gb->mlp_norm_gamma, gb->mlp_norm_beta,
                        seq_len, dim);

    // skip connection: d(mlp_in) = dresidual (from add) + dx_from_ln
    for (size_t i = 0; i < seq_len * dim; i++) {
        rt->dresidual[i] += rt->dscratch2[i];
    }
}

static void attention_backward(const struct transformer_block *b, struct grads_block *gb,
                               struct runtime *rt, size_t layer, size_t num_heads) {
    size_t seq_len = rt->seq_len;
    size_t dim = rt->dim;
    size_t head_dim = dim / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t max_seq = rt->max_seq_len;

    /*
      residual = attn_in + attn_update
      d(attn_update) starts as dresidual; skip path keeps dresidual for attn_in too.
    */

    // attn_update = head_outputs @ W_o
    // d_head -> dscratch, dW_o accumulate
    matmul_backward(rt->head_outputs[layer], b->attn_output_weights, rt->dresidual,
                    rt->dscratch, gb->attn_output_weights,
                    seq_len, dim, dim, 0, 1);

    // dscratch holds d_head_outputs. Backprop updates dQ/dK/dV.
    // dscratch2 used as dQ accumulator workspace... we need dQ, dK, dV buffers.

    float *d_head = rt->dscratch;
    float *dQ = rt->dscratch2;
    float *dK = rt->dhidden;
    float *dV = rt->dhidden + seq_len * dim;
    zero_n(dQ, seq_len * dim);
    zero_n(dK, seq_len * dim);
    zero_n(dV, seq_len * dim);

    for (size_t head = 0; head < num_heads; head++) {
        size_t head_offset = head * head_dim;

        for (size_t q_idx = 0; q_idx < seq_len; q_idx++) {
            const float *prob_row = rt->attn_probs[layer] +
                (head * max_seq + q_idx) * max_seq;

            // dw_k = sum_col d_head[q, col] * V[k, col]
            // and dV[k] += w_k * d_head[q]
            for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                float dw = 0.0f;
                for (size_t col = 0; col < head_dim; col++) {
                    float dh = d_head[q_idx * dim + head_offset + col];
                    dw += dh * rt->V[layer][k_idx * dim + head_offset + col];
                    dV[k_idx * dim + head_offset + col] +=
                        prob_row[k_idx] * dh;
                }
                rt->attn_weights[k_idx] = dw; // reuse as dw buffer
            }

            // softmax backward: ds_i = p_i * (dw_i - sum p_j dw_j)
            float sum_p_dw = 0.0f;
            for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                sum_p_dw += prob_row[k_idx] * rt->attn_weights[k_idx];
            }
            for (size_t k_idx = 0; k_idx <= q_idx; k_idx++) {
                float ds = prob_row[k_idx] * (rt->attn_weights[k_idx] - sum_p_dw);
                // scores = scale * Q·K  => dQ += scale*ds*K, dK += scale*ds*Q
                for (size_t col = 0; col < head_dim; col++) {
                    dQ[q_idx * dim + head_offset + col] +=
                        scale * ds * rt->K[layer][k_idx * dim + head_offset + col];
                    dK[k_idx * dim + head_offset + col] +=
                        scale * ds * rt->Q[layer][q_idx * dim + head_offset + col];
                }
            }
        }
    }

    float *d_normed = d_head; // reuse — d_head no longer needed
    zero_n(d_normed, seq_len * dim);

    matmul_backward(rt->attn_normed[layer], b->query_weights, dQ,
                    d_normed, gb->query_weights,
                    seq_len, dim, dim, 1, 1);
    matmul_backward(rt->attn_normed[layer], b->key_weights, dK,
                    d_normed, gb->key_weights,
                    seq_len, dim, dim, 1, 1);
    matmul_backward(rt->attn_normed[layer], b->value_weights, dV,
                    d_normed, gb->value_weights,
                    seq_len, dim, dim, 1, 1);

    // LN backward into temp + skip
    // using dQ as dx from LN
    layer_norm_backward(rt->attn_in[layer], d_normed,
                        b->attn_norm_gamma,
                        dQ, gb->attn_norm_gamma, gb->attn_norm_beta,
                        seq_len, dim);

    for (size_t i = 0; i < seq_len * dim; i++) {
        rt->dresidual[i] += dQ[i];
    }
}

int model_backward(const struct transformer *t, struct runtime *rt,
                   struct grads *g, const int *token_ids) {
    (void)token_ids; // targets already in rt->dlogits
    if (t == NULL || rt == NULL || g == NULL) return -1;

    size_t seq_len = rt->seq_len;
    size_t dim = t->dim;
    size_t vocab = t->vocab_size;

    /*
      Start from dlogits and walk backward:
      unembed -> final LN -> layers (MLP then attn) -> embeddings
    */

    // logits = final_normed @ W_unembed
    matmul_backward(rt->final_normed, t->unembedding_weights, rt->dlogits,
                    rt->dscratch, g->unembedding_weights,
                    seq_len, dim, vocab, 0, 1);

    // final LN: x is residual after last block (= layer_residuals[last] or residual)
    // After forward, rt->residual still holds the post-last-block state
    layer_norm_backward(rt->residual, rt->dscratch,
                        t->final_norm_gamma,
                        rt->dresidual, g->final_norm_gamma, g->final_norm_beta,
                        seq_len, dim);

    for (size_t layer = t->num_layers; layer-- > 0; ) {
        mlp_backward(&t->blocks[layer], &g->blocks[layer], rt, layer);
        attention_backward(&t->blocks[layer], &g->blocks[layer], rt, layer, t->num_heads);
    }

    // token_ids[i]=3 means row 3 of the token embedding table gets that residual grad.
    for (size_t i = 0; i < seq_len; i++) {
        int tid = token_ids[i];
        for (size_t j = 0; j < dim; j++) {
            float d = rt->dresidual[i * dim + j];
            g->token_embeddings[tid * dim + j] += d;
            g->position_embeddings[i * dim + j] += d;
        }
    }

    return 0;
}

void sgd_step(struct transformer *t, const struct grads *g, float lr) {
    size_t dim = t->dim;
    size_t vocab = t->vocab_size;
    size_t ctx = t->max_context;

    sgd_buf(t->token_embeddings, g->token_embeddings, vocab * dim, lr);
    sgd_buf(t->position_embeddings, g->position_embeddings, ctx * dim, lr);
    sgd_buf(t->final_norm_gamma, g->final_norm_gamma, dim, lr);
    sgd_buf(t->final_norm_beta, g->final_norm_beta, dim, lr);
    sgd_buf(t->unembedding_weights, g->unembedding_weights, dim * vocab, lr);

    for (size_t l = 0; l < t->num_layers; l++) {
        struct transformer_block *b = &t->blocks[l];
        struct grads_block *gb = &g->blocks[l];
        sgd_buf(b->query_weights, gb->query_weights, dim * dim, lr);
        sgd_buf(b->key_weights, gb->key_weights, dim * dim, lr);
        sgd_buf(b->value_weights, gb->value_weights, dim * dim, lr);
        sgd_buf(b->attn_output_weights, gb->attn_output_weights, dim * dim, lr);
        sgd_buf(b->attn_norm_gamma, gb->attn_norm_gamma, dim, lr);
        sgd_buf(b->attn_norm_beta, gb->attn_norm_beta, dim, lr);
        sgd_buf(b->mlp_norm_gamma, gb->mlp_norm_gamma, dim, lr);
        sgd_buf(b->mlp_norm_beta, gb->mlp_norm_beta, dim, lr);
        sgd_buf(b->mlp_up_weights, gb->mlp_up_weights, dim * 4 * dim, lr);
        sgd_buf(b->mlp_down_weights, gb->mlp_down_weights, 4 * dim * dim, lr);
    }
}

float train_step(struct transformer *t, struct runtime *rt, struct grads *g,
                 const int *token_ids, size_t seq_len, float lr) {
    if (seq_len < 2) return -1.0f;

    if (model_forward(t, rt, token_ids, seq_len) != 0) return -1.0f;

    float loss = cross_entropy_next_token(rt, token_ids);
    if (loss < 0.0f) return -1.0f;

    grads_zero(g);
    if (model_backward(t, rt, g, token_ids) != 0) return -1.0f;
    sgd_step(t, g, lr);

    return loss;
}
