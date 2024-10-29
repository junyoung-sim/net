#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../common/net.h"

void test() {
    srand(time(NULL));

    Net *linear  = make_net(100, 100, 5, LINEAR , 10);
    Net *relu    = make_net(100, 100, 5, RELU   , 10);
    Net *sigmoid = make_net(100, 100, 5, SIGMOID, 10);
    Net *softmax = make_net(100, 100, 5, SOFTMAX, 10);

    Vec *x = make_vec(100, 0.0f);
    for(int i = 0; i < x->size; i++) {
        x->dat[i] = rand_normal();
    }

    dump_vec(x);

    Vec *out = make_vec(5, 0.0f);

    printf("\nlinear\n");
    forward(linear, x, out);
    dump_vec(out);

    printf("\nrelu\n");
    forward(relu, x, out);
    dump_vec(out);

    printf("\nsigmoid\n");
    forward(sigmoid, x, out);
    dump_vec(out);

    printf("\nsoftmax\n");
    forward(softmax, x, out);
    dump_vec(out);

    free_net(linear);
    free_net(relu);
    free_net(sigmoid);
    free_net(softmax);

    free_vec(x);
    free_vec(out);
}

int main()
{
    test();

    return 0;
}