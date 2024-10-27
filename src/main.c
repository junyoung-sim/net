#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../common/net.h"

void test() {
    Net *net = make_net(3, 3, 1, LINEAR, 2);

    Vec *in = make_vec(3, 0.0f);
    in->dat[0] = (float)rand() / RAND_MAX;
    in->dat[1] = (float)rand() / RAND_MAX;
    in->dat[2] = (float)rand() / RAND_MAX;

    Vec *out = make_vec(1, 0.0f);

    forward(net, in, out);

    free_net(net);
    free_vec(in);
    free_vec(out);
}

int main()
{
    test();

    return 0;
}