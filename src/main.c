#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../common/net.h"

void test0() {
    Vec *vec = make_vec(3, 0.0f);
    vec->dat[0] = 1.0f;
    vec->dat[1] = 2.0f;
    vec->dat[2] = 3.0f;

    dump_vec(vec);
    free_vec(vec);

    Mat *mat = make_mat(2, 2, 0.0f);
    mat->dat[0][0] = 0.0f;
    mat->dat[0][1] = 1.0f;
    mat->dat[1][0] = 2.0f;
    mat->dat[1][1] = 3.0f;

    dump_mat(mat);
    free_mat(mat);
}

void test1() {
    srand(time(NULL));

    Mat *mat = make_mat(3, 3, 0.0f);
    for(int i = 0; i < mat->size[0]; i++) {
        for(int j = 0; j < mat->size[1]; j++) {
            mat->dat[i][j] = (float)rand() / RAND_MAX;
        }
    }

    Vec *vec = make_vec(3, 0.0f);
    for(int i = 0; i < vec->size; i++) {
        vec->dat[i] = (float)rand() / RAND_MAX;
    }

    Vec *prod = make_vec(3, 0.0f);
    mat_vec_product(mat, vec, prod);

    dump_mat(mat);
    dump_vec(vec);
    dump_vec(prod);

    free_mat(mat);
    free_vec(vec);
    free_vec(prod);
}

void test2() {
    Vec *a = make_vec(3, 0.0f);
    Vec *b = make_vec(3, 1.0f);

    vec_sum(a, b);
    dump_vec(a);

    free_vec(a);
    free_vec(b);
}

void test3() {
    Net *net = make_net(5, 5, 3, LINEAR, 3);

    dump_mat(net->grad[0]);
    dump_mat(net->grad[1]);
    dump_mat(net->grad[2]);

    dump_mat(net->weight[0]);
    dump_mat(net->weight[1]);
    dump_mat(net->weight[2]);

    free_net(net);
}

int main()
{
    //test0();

    //test1();

    //test2();

    test3();

    return 0;
}