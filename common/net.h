#ifndef __NET_H_
#define __NET_H_

#include "mat.h"

#define LINEAR  0
#define RELU    1
#define SIGMOID 2
#define SOFTMAX 3

void relu(Vec *x, Vec *out);

typedef struct Net Net;
struct Net
{
    int input_size;
    int hidden_size;
    int output_size;
    int output_type;
    int num_of_layers;

    Mat **grad;
    Mat **weight;
    Vec **bias;
    Vec **sum;
    Vec **act;
    Vec **err;
};

Net *make_net(
    int input_size,
    int hidden_size,
    int output_size,
    int output_type,
    int num_of_layers
);

void forward(Net *net, Vec *x, Vec *out);

void free_net(Net *net);

#endif