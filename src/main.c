#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../common/net.h"

int main()
{
    srand(time(NULL));
    
    int shape[4] = {10, 8, 6, 4};

    Net *net = make_net(shape, 4, 10, SOFTMAX);

    Vec *x = make_vec(10, 0.0f);
    for(int i = 0; i < x->size; i++) {
        x->dat[i] = rand_normal();
    }

    Vec *out = make_vec(4, 0.0f);
    out = forward(net, x);

    dump_vec(out);

    free_net(net);
    free_vec(x);
    free_vec(out);

    return 0;
}