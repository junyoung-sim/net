#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../common/net.h"

int main()
{
    srand(time(NULL));
    
    int INPUT_SIZE    = 100;
    int OUTPUT_SIZE   = 5;
    int NUM_OF_LAYERS = 5;
    int BATCH_SIZE    = 1;

    int shape[NUM_OF_LAYERS];
    shape[0] = 100;
    shape[1] = 100;
    shape[2] = 100;
    shape[3] = 100;
    shape[4] = OUTPUT_SIZE;

    Net* net = make_net(shape, NUM_OF_LAYERS, INPUT_SIZE, SOFTMAX);

    Vec** x = calloc(BATCH_SIZE, sizeof(Vec*));
    Vec** y = calloc(BATCH_SIZE, sizeof(Vec*));

    for(int i = 0; i < BATCH_SIZE; i++) {
        x[i] = make_vec(INPUT_SIZE,  0.0f);
        y[i] = make_vec(OUTPUT_SIZE, 0.0f);
        for(int j = 0; j < INPUT_SIZE; j++) {
            x[i]->dat[j] = rand_normal();
        }
        y[i]->dat[rand() % OUTPUT_SIZE] = 1.0f;
    }

    for(int t = 0; t < 10; t++) {
        zero_grad(net);
        for(int i = 0; i < BATCH_SIZE; i++) {
            backward(net, x[i], y[i], 0.001f, 0.001f);
        }
        step(net);

        Vec* yhat = make_vec(OUTPUT_SIZE, 0.0f);

        float batch_loss = 0.0f;
        for(int i = 0; i < BATCH_SIZE; i++) {
            forward(net, x[i], yhat);
            float loss = 0.0f;
            for(int j = 0; j < OUTPUT_SIZE; j++) {
                loss += -1.0f * y[i]->dat[j] * log(yhat->dat[j]);
            }
            batch_loss += loss;
        }
        batch_loss /= BATCH_SIZE;

        printf("t=%d L=%f\n", t, batch_loss);

        free_vec(yhat);
    }

    //---------------------------------------------------------------

    free_net(net);

    for(int i = 0; i < BATCH_SIZE; i++) {
        free_vec(x[i]);
        free_vec(y[i]);
    }
    free(x);
    free(y);

    return 0;
}