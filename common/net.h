#ifndef __NET_H_
#define __NET_H_

#include "mat.h"

#define PI 3.141593

#define LINEAR  0
#define SIGMOID 1
#define SOFTMAX 2

float rand_normal();

void linear(Vec *x, Vec *out);
void relu(Vec *x, Vec *out);
void sigmoid(Vec *x, Vec *out);
void softmax(Vec *x, Vec *out);

typedef struct Net Net;
struct Net
{
    int *shape;
    int input_size;
    int output_type;
    int num_of_layers;

    Mat **grad;
    Mat **weight;
    Vec **bias;
    Vec **sum;
    Vec **act;
    Vec **err;
};

Net *make_net(int* shape, int num_of_layers, int input_size, int output_type);

Vec* forward(Net *net, Vec *x);

void backward(Net *net, Vec *x, Vec *y, float alpha, float lambda);

void free_net(Net *net);

#endif