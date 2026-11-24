#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "helpers.h"
#include "../include/model.h"

int main(void) {

    float a[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    float b[4] = { 2.0f, 4.0f, 2.0f, 4.0f };
    float c[4];

    matprint(a, 2, 2);
    printf("\n*\n");
    matprint(b, 2, 2);
    printf("\n=\n");
    matmul(a, b, c, 2, 2, 2);
    matprint(c, 2, 2);

    float z[8] = { 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f };
    float p[8];

    softmax(z, p, 8);
    matprint(z, 1, 8);
    matprint(p, 1, 8);

    srand(42);

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

    int token_ids[] = {3, 1, 4};
    size_t seq_len = 3;

    if (model_forward(&t, &rt, token_ids, seq_len) != 0) {
        printf("model_forward failed\n");
        runtime_free(&rt);
        model_free(&t);
        return 1;
    }

    printf("\nlogits (seq_len x vocab_size):\n");
    matprint(rt.logits, seq_len, vocab_size);

    printf("\nnext-token probs:\n");
    matprint(rt.probs, 1, vocab_size);

    size_t predicted = 0;
    for (size_t i = 1; i < vocab_size; i++)
        if (rt.probs[i] > rt.probs[predicted]) predicted = i;
    printf("predicted next token id: %zu\n", predicted);

    runtime_free(&rt);
    model_free(&t);

    return 0;
}
