#ifndef TRAIN_H
#define TRAIN_H

#include "model.h"

/*
  grads just mirror transformer weights
*/
struct grads_block {
    float *query_weights;
    float *key_weights;
    float *value_weights;
    float *attn_output_weights;
    float *attn_norm_gamma;
    float *attn_norm_beta;
    float *mlp_norm_gamma;
    float *mlp_norm_beta;
    float *mlp_up_weights;
    float *mlp_down_weights;
};

struct grads {
    size_t vocab_size, dim, max_context, num_layers;
    float *token_embeddings;      // vocab × dim
    float *position_embeddings;   // max_context × dim
    float *final_norm_gamma;
    float *final_norm_beta;
    float *unembedding_weights;   // dim × vocab
    struct grads_block *blocks;
};

int grads_init(struct grads *g, const struct transformer *t);
void grads_free(struct grads *g);
void grads_zero(struct grads *g);

/*
  next token cross entropy.

  token_ids = {3, 1, 4, 2}
   position 0's logits should select 1
   position 1's logits should select 4
   position 2's logits should select 2
  last token has no target

  also writes rt->dlogits = (softmax(logits) - onehot) / n_targets so backward starts there
*/
float cross_entropy_next_token(struct runtime *rt, const int *token_ids);

int model_backward(const struct transformer *t, struct runtime *rt,
                   struct grads *g, const int *token_ids);

// w -= lr * dw  for every weight tensor
void sgd_step(struct transformer *t, const struct grads *g, float lr);

/*
  training step: forward -> loss -> zero grads -> backward -> sgd
  returns loss, or -1.0f (error)
  seq_len must be >= 2
*/
float train_step(struct transformer *t, struct runtime *rt, struct grads *g,
                 const int *token_ids, size_t seq_len, float lr);

#endif // TRAIN_H
