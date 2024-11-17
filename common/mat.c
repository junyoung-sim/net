#include <stdio.h>
#include <stdlib.h>

#include "mat.h"

Vec* make_vec(int size, float init) {
    Vec* vec = calloc(1, sizeof(Vec));
    vec->size = size;
    vec->dat = calloc(size, sizeof(float));
    for(int i = 0; i < size; i++) {
        vec->dat[i] = init;
    }
    return vec;
}

Mat* make_mat(int rows, int cols, float init) {
    Mat* mat = calloc(1, sizeof(Mat));
    mat->size[0] = rows;
    mat->size[1] = cols;
    mat->dat = calloc(rows, sizeof(float*));
    for(int i = 0; i < rows; i++) {
        mat->dat[i] = calloc(cols, sizeof(float));
        for(int j = 0; j < cols; j++) {
            mat->dat[i][j] = init;
        }
    }
    return mat;
}

void mat_vec_product(Mat* mat, Vec* vec, Vec* out) {
    for(int i = 0; i < mat->size[0]; i++) {
        for(int j = 0; j < mat->size[1]; j++) {
            out->dat[i] += mat->dat[i][j] * vec->dat[j];
        }
    }
}

void vec_sum(Vec* vec, Vec* diff) {
    for(int i = 0; i < vec->size; i++) {
        vec->dat[i] += diff->dat[i];
    }
}

void dump_vec(Vec* vec) {
    printf("vec([");
    for(int i = 0; i < vec->size; i++) {
        printf("%f", vec->dat[i]);
        printf((i != vec->size - 1) ? " " : "] ");
    }
    printf("size=%d)\n", vec->size);
}

void dump_mat(Mat* mat) {
    printf("mat([");
    for(int i = 0; i < mat->size[0]; i++) {
        printf((i == 0) ? "" : "     ");
        printf("[");
        for(int j = 0; j < mat->size[1]; j++) {
            printf("%f", mat->dat[i][j]);
            printf((j != mat->size[1] - 1) ? " " : "]");
        }
        printf((i != mat->size[0] - 1) ? "\n" : "");
    }
    printf("] size=%dx%d)\n", mat->size[0], mat->size[1]);
}

void free_vec(Vec* vec) {
    free(vec->dat);
    free(vec);
}

void free_mat(Mat* mat) {
    for(int i = 0; i < mat->size[0]; i++) {
        free(mat->dat[i]);
    }
    free(mat->dat);
    free(mat);
}