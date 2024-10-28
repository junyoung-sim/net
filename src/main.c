#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../common/net.h"

void test() {
    srand(time(NULL));

    Net *linear  = make_net(10, 10, 3, LINEAR , 3);
    Net *relu    = make_net(10, 10, 3, RELU   , 3);
    Net *sigmoid = make_net(10, 10, 3, SIGMOID, 3);
    Net *softmax = make_net(10, 10, 3, SOFTMAX, 3);

    Vec *x = make_vec(10, 0.0f);
    for(int i = 0; i < x->size; i++) {
        x->dat[i] = rand_normal();
    }

    dump_vec(x);

    Vec *out = make_vec(3, 0.0f);

    printf("linear\n");
    forward(linear, x, out);
    dump_vec(linear->act[2]);

    printf("\nrelu\n");
    forward(relu, x, out);
    dump_vec(relu->act[2]);

    printf("\nsigmoid\n");
    forward(sigmoid, x, out);
    dump_vec(sigmoid->act[2]);

    printf("\nsoftmax\n");
    forward(softmax, x, out);
    dump_vec(softmax->act[2]);

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