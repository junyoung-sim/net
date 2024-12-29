#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/net.h"

int main(int argc, char *argv[])
{
  srand(time(NULL));
  printf("\n");

  //==========================================================
  // Generate Random I/O
  //==========================================================

  Vec* x = make_vec(10000, 0.0f);
  for(int i = 0; i < 10000; i++) {
    x->dat[i] = rand_normal();
  }

  Vec* y = make_vec(10, 0.0f);
  y->dat[rand() % 10] = 1.0f;

  //==========================================================
  // Model Initialization (1B+)
  //==========================================================

  int shape[10];
  shape[0] = 10000;
  shape[1] = 10000;
  shape[2] = 10000;
  shape[3] = 10000;
  shape[4] = 10000;
  shape[5] = 10000;
  shape[6] = 10000;
  shape[7] = 10000;
  shape[8] = 10000;
  shape[9] = 10;

  Net* net;
  
  clock_t init_t0 = clock();
  {
    net = make_net(shape, 10, 10000, SOFTMAX);
  }
  clock_t init_tf = clock();

  //==========================================================
  // Forward Pass
  //==========================================================

  Vec* yhat = make_vec(10, 0.0f);

  clock_t forward_t0 = clock();
  {
    forward(net, x, yhat);
    dump_vec(yhat);
  }
  clock_t forward_tf = clock();

  //==========================================================
  // Backward Pass
  //==========================================================



  //==========================================================
  // Report Timing
  //==========================================================

  clock_t init_dt_c = init_tf - init_t0;
  float   init_dt_s = (float)init_dt_c / CLOCKS_PER_SEC;

  clock_t forward_dt_c = forward_tf - forward_t0;
  float   forward_dt_s = (float)forward_dt_c / CLOCKS_PER_SEC;

  printf("\n");
  printf("Model Init. Timing  (s): %f\n", init_dt_s);
  printf("Forward Pass Timing (s): %f\n", forward_dt_s);
  printf("\n");

  //==========================================================
  // Free Memory
  //==========================================================

  free_vec(x);
  free_vec(y);
  free_vec(yhat);

  free_net(net);

  return 0;

}