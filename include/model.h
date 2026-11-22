
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
    float *attn_output_weights;// dim × dim   (projects attention result back into the residual stream (modified input mat)
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


//randomly initialize the embeddings to floats within some small range centered around 0

int model_init(struct transformer *t, size_t vocab_size, size_t dim,
    size_t max_context, size_t num_layers, size_t num_heads);

float *model_forward(struct transformer *t, float *input_mat, size_t seq_len);

float *get_input_embeddings(const struct transformer *t, const int *token_ids, size_t seq_len);

int model_save(const struct transformer *t, const char *path);

void model_free(struct transformer *t);


#endif //MODEL_H