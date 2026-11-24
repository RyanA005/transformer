#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

struct transformer {
    size_t vocab_size, dim, max_context, num_layers, num_heads;
    float *token_embeddings;       // vocab × dim
    float *position_embeddings;    // max_context × dim
    struct transformer_block *blocks; // num_layers of them
    float *final_norm_gamma;   // dim  (before unembedding)
    float *final_norm_beta;    // dim
    float *unembedding_weights;    // dim × vocab  (final projection to logits)
};

struct transformer_block {
    // attention
    float *query_weights;      // dim × dim
    float *key_weights;        // dim × dim
    float *value_weights;      // dim × dim
    float *attn_output_weights;// dim × dim   (projects attention result back into the residual stream)
    float *attn_norm_gamma;    // dim  (scale, init 1)
    float *attn_norm_beta;     // dim  (shift, init 0)
    // pre-MLP LayerNorm
    float *mlp_norm_gamma;     // dim
    float *mlp_norm_beta;      // dim
    // MLP
    float *mlp_up_weights;     // d × 4d  (expand)
    float *mlp_down_weights;   // 4d × d  (contract back)

    // might change name of gamma and beta directly to scale and shift across codebase not sure,
    // ts kinda confusing as is
};

/*
  Preallocated forward-pass activations + caches for backward.
*/
struct runtime {
    size_t max_seq_len;   // transformer.max_context
    size_t dim;
    size_t vocab_size;
    size_t num_layers;
    size_t num_heads;
    size_t seq_len;       // length used by last forward

    float *residual;              // max_seq × dim  (live residual stream)
    float **layer_residuals;      // [num_layers] post-block residual snapshot

    // per-layer attention caches
    float **attn_in;              // residual before attn LN
    float **attn_normed;          // after attn LN
    float **Q, **K, **V;
    float **head_outputs;
    float **attn_update;
    float **attn_probs;

    // per-layer MLP caches
    float **mlp_in;               // residual before mlp LN
    float **mlp_normed;
    float **mlp_pre_gelu;         // after up-proj, before GELU
    float **mlp_hidden;           // after GELU
    float **mlp_post_update;      // down-proj output added to residual

    float *final_normed;          // after final LN, before unembed
    float *logits;                // max_seq × vocab
    float *probs;                 // vocab (last-token next-token probs, for inference print)
    float *dlogits;               // max_seq × vocab (filled by loss)

    // scratch reused during forward
    float *scores;                // max_seq
    float *attn_weights;          // max_seq (one query's softmax)
    float *dresidual;             // max_seq × dim  (grad flowing through residual)
    float *dscratch;              // max_seq × dim
    float *dscratch2;             // max_seq × dim
    float *dhidden;               // max_seq × (4·dim)
};


int model_init(struct transformer *t, size_t vocab_size, size_t dim,
    size_t max_context, size_t num_layers, size_t num_heads);

int runtime_init(struct runtime *rt, const struct transformer *t);

int model_forward(const struct transformer *t, struct runtime *rt,
                  const int *token_ids, size_t seq_len);

int model_save(const struct transformer *t, const char *path);

void model_free(struct transformer *t);

void runtime_free(struct runtime *rt);

#endif //MODEL_H
