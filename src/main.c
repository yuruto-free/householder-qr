#include "householder_qr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#define MAX_TEST_DATA (3)
#define MAX_ITER (1000)

struct data_t {
    int32_t dim;
    double *A;
    double *eigen_val;
};
double matrix1[9] = {
    2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 5.0, 3.0,
};
double lambda1[3] = {
    5.0, 2.0, 0.0,
};
double matrix2[16] = {
    1.0,  2.0, 3.0,   4.0,
    0.0,  2.0, 0.0,   5.0,
    0.0,  3.0, 3.0, -15.0,
    0.0, 11.0, 1.0,   5.0,
};
double lambda2[4] = {
    10.0, 3.0, -3.0, 1.0,
};
double matrix3[25] = {
    1.0, -1.0,  -2.0,  2.0,  6.0,
    0.0,  0.0,   0.0,  0.0,  2.0,
    0.0, -1.0,   5.0, -5.0, -1.0,
    0.0, -0.8,  12.0,  1.0, -2.0,
    0.0,  2.0, -38.0,  0.0,  8.0,
};
double lambda3[5] = {
    5.0, 4.0, 3.0, 2.0, 1.0,
};
struct data_t TEST_DATA[MAX_TEST_DATA] = {
    {3, matrix1, lambda1},
    {4, matrix2, lambda2},
    {5, matrix3, lambda3},
};

static void print_vec(int32_t dim, double *exact, double *estimated) {
    int32_t i;

    if ((NULL != exact) && (NULL != estimated)) {
        printf("estimated:");
        for (i = 0; i < dim; i++) {
            printf(" %+.5f", estimated[i]);
        }
        printf("\n");
        printf("exact:    ");
        for (i = 0; i < dim; i++) {
            printf(" %+.5f", exact[i]);
        }
        printf("\n");
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int32_t dim;
    int32_t func_val;
    double *lambda;
    struct data_t *target;
    int i;

    for (i = 0; i < (int)MAX_TEST_DATA; i++) {
        target = &TEST_DATA[i];
        dim = target->dim;
        lambda = (double *)malloc(sizeof(double) * dim);

        if (NULL != lambda) {
            func_val = QR_method(dim, target->A, lambda, (int32_t)MAX_ITER);

            if ((int32_t)QR_OK == func_val) {
                print_vec(dim, target->eigen_val, lambda);
            }
            free(lambda);
        }
    }   

    return 0;
}