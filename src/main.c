#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "helpers.h"
#include "../include/model.h"
#include "../include/train.h"

int main(void) {
    srand(1234);

    /*
      quick overfit demo, we are memorizing a sequence just to make sure loss drops

      Ex: tokens {3,1,4,2} means:
        after seeing 3, predict 1
        after seeing 3,1 predict 4
        after seeing 3,1,4 predict 2
    */
    size_t vocab_size = 8, dim = 8, max_context = 16, num_layers = 1, num_heads = 2;
    struct transformer t;
    if (model_init(&t, vocab_size, dim, max_context, num_layers, num_heads) != 0) {
        printf("model_init failed\n");
        return 1;
    }

    struct runtime rt;
    if (runtime_init(&rt, &t) != 0) {
        printf("runtime_init failed\n");
        model_free(&t);
        return 1;
    }

    struct grads g;
    if (grads_init(&g, &t) != 0) {
        printf("grads_init failed\n");
        runtime_free(&rt);
        model_free(&t);
        return 1;
    }

    int token_ids[] = {3, 1, 4, 2};
    size_t seq_len = 4;
    float lr = 0.05f;
    int steps = 200;

    printf("training on [3, 1, 4, 2] for %d steps (lr=%.3f)\n", steps, lr);

    float first_loss = -1.0f;
    float last_loss = -1.0f;
    for (int step = 0; step < steps; step++) {
        float loss = train_step(&t, &rt, &g, token_ids, seq_len, lr);
        if (loss < 0.0f) {
            printf("train_step failed at step %d\n", step);
            grads_free(&g);
            runtime_free(&rt);
            model_free(&t);
            return 1;
        }
        if (step == 0) first_loss = loss;
        last_loss = loss;
        if (step % 20 == 0 || step == steps - 1) {
            printf("step %3d  loss=%.6f\n", step, loss);
        }
    }

    int prefix[] = {3, 1, 4};
    if (model_forward(&t, &rt, prefix, 3) != 0) {
        printf("final model_forward failed\n");
        grads_free(&g);
        runtime_free(&rt);
        model_free(&t);
        return 1;
    }

    printf("\nprobs after training, prefix [3,1,4], target 2:\n");
    matprint(rt.probs, 1, vocab_size);

    size_t predicted = 0;
    for (size_t i = 1; i < vocab_size; i++)
        if (rt.probs[i] > rt.probs[predicted]) predicted = i;
    printf("predicted token id: %zu\n", predicted);
    printf("\nloss %.6f -> %.6f\n", first_loss, last_loss);

    grads_free(&g);
    runtime_free(&rt);
    model_free(&t);

    return 0;
}
