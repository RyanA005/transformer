#include <stdio.h>
#include <stdint.h>

#include "helpers.h"

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


    return 0;
}
