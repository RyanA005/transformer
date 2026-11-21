
#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

struct transformer {
    size_t vocab_size, model_dim, max_context, num_layers;
    float *token_embeddings;       // vocab × model_dim
    float *position_embeddings;    // max_context × model_dim
    struct transformer_block *blocks; // num_layers of them
    float *unembedding_weights;    // model_dim × vocab  (final projection to logits)
};

struct transformer_block {
    // attention
    float *query_weights;      // d × d
    float *key_weights;        // d × d
    float *value_weights;      // d × d
    float *attn_output_weights;// d × d   (projects attention result back into the residual stream)
    // MLP
    float *mlp_up_weights;     // d × 4d  (expand)
    float *mlp_down_weights;   // 4d × d  (contract back)
};


//randomly initialize the embeddings to floats within some small range centered around 0

int model_init(struct transformer *t, size_t vocab_size, size_t dim, size_t max_context, size_t num_layers);

float *model_forward(struct transformer *t, float *input_embeds, size_t seq_len);

float *get_input_embeddings(const struct transformer *t, const int *token_ids, size_t seq_len);

int model_save(const struct transformer *t, const char *path);

void model_free(struct transformer *t);


#endif //MODEL_H