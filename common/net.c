#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "net.h"

//==========================================================
// Operations
//==========================================================

float rand_normal() {
  
  float r1 = (float)(rand() + 0.5f) / (RAND_MAX + 1.0f);
  float r2 = (float)(rand() + 0.5f) / (RAND_MAX + 1.0f);

  return sqrt(-2.0f * log(r1)) * sin(2.0f * PI * r2);

}

void linear
(
  Vec* x,
  Vec* out
) {
  
  for(int i = 0; i < x->size; i++) {
    out->dat[i] = x->dat[i];
  }

}

void relu
(
  Vec* x,
  Vec* out
) {

  for(int i = 0; i < x->size; i++) {
    out->dat[i] = (x->dat[i] <= 0.0f ? 0.0f : x->dat[i]);
  }

}

void sigmoid
(
  Vec* x,
  Vec* out
) {

  for(int i = 0; i < x->size; i++) {
    out->dat[i] = 1.0f / (1.0f + exp(-x->dat[i]));
  }

}

void softmax
(
  Vec* x,
  Vec* out
) {

  float norm = 0.0f;
  for(int i = 0; i < x->size; i++) {
    norm += exp(x->dat[i]);
  }

  for(int i = 0; i < x->size; i++) {
    out->dat[i] = exp(x->dat[i]) / norm;
  }

}

float drelu
(
  float x
) {

  return (x <= 0.0f ? 0.0f : 1.0f);

}

float cross_entropy
(
  Vec* y,
  Vec* yhat
) {

  float loss = 0.0f;
  for(int i = 0; i < y->size; i++) {
    loss += -y->dat[i] * log(yhat->dat[i]);
  }

  return loss;

}

//==========================================================
// Neural Network
//==========================================================

Net* make_net
(
  int *shape,
  int num_of_layers,
  int input_size,
  int output_type
) {

  Net* net = (Net*)malloc(sizeof(Net));

  net->shape         = shape;
  net->input_size    = input_size;
  net->output_type   = output_type;
  net->num_of_layers = num_of_layers;

  net->grad   = (Mat**)malloc(num_of_layers * sizeof(Mat*));
  net->weight = (Mat**)malloc(num_of_layers * sizeof(Mat*));
  net->bias   = (Vec**)malloc(num_of_layers * sizeof(Vec*));
  net->sum    = (Vec**)malloc(num_of_layers * sizeof(Vec*));
  net->act    = (Vec**)malloc(num_of_layers * sizeof(Vec*));
  net->err    = (Vec**)malloc(num_of_layers * sizeof(Vec*));

  net->backward_count = 0;

  for(int l = 0; l < num_of_layers; l++) {

    int out = shape[l];
    int in  = (l == 0 ? input_size : shape[l-1]);

    net->grad[l]   = make_mat(out, in+1, 0.0f);
    net->weight[l] = make_mat(out, in,   0.0f);

    net->bias[l] = make_vec(out, 0.0f);
    net->sum[l]  = make_vec(out, 0.0f);
    net->act[l]  = make_vec(out, 0.0f);
    net->err[l]  = make_vec(out, 0.0f);

  }

  for(int l = 0; l < num_of_layers; l++) {

    int out = shape[l];
    int in  = (l == 0 ? input_size : shape[l-1]);

    for(int n = 0; n < out; n++) {
      for(int i = 0; i < in; i++) {
        float scale = sqrt(2.0f / in);
        net->weight[l]->dat[n][i] = scale * rand_normal();
      }
    }

  }

  return net;

}

void forward
(
  Net* net,
  Vec* x,
  Vec* out
) {

  for(int l = 0; l < net->num_of_layers; l++) {

    int out = net->shape[l];
    for(int n = 0; n < out; n++) {
      net->sum[l]->dat[n] = 0.0f;
      net->act[l]->dat[n] = 0.0f;
      net->err[l]->dat[n] = 0.0f;
    }

    mat_vec_product
    (
      net->weight[l],
      (l == 0 ? x : net->act[l-1]),
      net->sum[l]
    );
    vec_sum(net->sum[l], net->bias[l]);

    if(l == net->num_of_layers - 1) continue;

    relu(net->sum[l], net->act[l]);

  }

  int lout = net->num_of_layers - 1;

  switch(net->output_type) {
    case LINEAR  : linear (net->sum[lout], net->act[lout]);
    case SIGMOID : sigmoid(net->sum[lout], net->act[lout]);
    case SOFTMAX : softmax(net->sum[lout], net->act[lout]); 
  }

  for(int i = 0; i < out->size; i++) {
    out->dat[i] = net->act[lout]->dat[i];
  }

}

void backward
(
  Net* net,
  Vec* x,
  Vec* y,
  float alpha,
  float lambda
) {

  Vec* yhat = make_vec(y->size, 0.0f);
  forward(net, x, yhat);

  for(int l = net->num_of_layers - 1; l >= 0; l--) {

    float agrad  = 0.0f;
    float sgrad  = 0.0f;
    float wgrad  = 0.0f;
    float weight = 0.0f;

    int out = net->shape[l];
    int in  = (l == 0 ? net->input_size : net->shape[l-1]);

    for(int n = 0; n < out; n++) {

      agrad = net->err[l]->dat[n] * drelu(net->sum[l]->dat[n]);

      if(l == net->num_of_layers - 1) {
        switch(net->output_type) {
          case LINEAR  : agrad = y->dat[n] - yhat->dat[n];
          case SIGMOID : agrad = yhat->dat[n] - y->dat[n];
          case SOFTMAX : agrad = yhat->dat[n] - y->dat[n];
        }
      }

      net->grad[l]->dat[n][in] += alpha * agrad;

      for(int i = 0; i < in; i++) {
        
        weight = net->weight[l]->dat[n][i];
        if(l != 0) {
          net->err[l-1]->dat[i] += agrad * weight;
        }

        sgrad = net->act[l-1]->dat[i];
        if(l == 0) {
          sgrad = x->dat[i];
        }

        wgrad  = agrad * sgrad;
        wgrad += lambda * weight;

        net->grad[l]->dat[n][i] += alpha * wgrad;

      }

    }

  }

  net->backward_count++;

}

void step
(
  Net* net
) {

  float bgrad = 0.0f;
  float wgrad = 0.0f;

  for(int l = 0; l < net->num_of_layers; l++) {

    int out = net->shape[l];
    int in  = (l == 0 ? net->input_size : net->shape[l-1]);

    for(int n = 0; n < out; n++) {

      bgrad = net->grad[l]->dat[n][in] / net->backward_count;
      net->bias[l]->dat[n] -= bgrad;

      for(int i = 0; i < in; i++) {
        wgrad = net->grad[l]->dat[n][i] / net->backward_count;
        net->weight[l]->dat[n][i] -= wgrad;
      }

    }

  }

}

void zero_grad
(
  Net* net
) {

  for(int l = 0; l < net->num_of_layers; l++) {

    int out = net->shape[l];
    int in  = (l == 0 ? net->input_size : net->shape[l-1]);

    for(int n = 0; n < out; n++) {
      for(int i = 0; i <= in; i++) {
        net->grad[l]->dat[n][i] = 0.0f;
      }
    }

  }

  net->backward_count = 0;

}

void free_net
(
  Net* net
) {

  for(int l = 0; l < net->num_of_layers; l++) {
    free_mat(net->grad[l]);
    free_mat(net->weight[l]);
    free_vec(net->bias[l]);
    free_vec(net->sum[l]);
    free_vec(net->act[l]);
    free_vec(net->err[l]);
  }

  free(net->grad);
  free(net->weight);
  free(net->bias);
  free(net->sum);
  free(net->act);
  free(net->err);
  free(net);

}