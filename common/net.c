#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "net.h"

void relu(Vec *x, Vec *out) {
    for(int i = 0; i < x->size; i++) {
        out->dat[i] = (x->dat[i] < 0.0f ? 0.0f : x->dat[i]);
    }
}

Net *make_net(
    int input_size,
    int hidden_size,
    int output_size,
    int output_type,
    int num_of_layers
) {
    Net *net = calloc(1, sizeof(Net));

    net->input_size    = input_size;
    net->hidden_size   = hidden_size;
    net->output_size   = output_size;
    net->output_type   = output_type;
    net->num_of_layers = num_of_layers;

    net->grad   = calloc(num_of_layers, sizeof(Mat*));
    net->weight = calloc(num_of_layers, sizeof(Mat*));
    net->bias   = calloc(num_of_layers, sizeof(Vec*));
    net->sum    = calloc(num_of_layers, sizeof(Vec*));
    net->act    = calloc(num_of_layers, sizeof(Vec*));
    net->err    = calloc(num_of_layers, sizeof(Vec*));

    for(int l = 0; l < num_of_layers; l++) {
        int n = (l != num_of_layers - 1 ? hidden_size : output_size);
        int i = (l == 0 ? input_size : hidden_size);

        net->grad[l]   = make_mat(n, i+1, 0.0f);
        net->weight[l] = make_mat(n, i, 0.0f);
        net->bias[l]   = make_vec(n, 0.0f);
        net->sum[l]    = make_vec(n, 0.0f);
        net->act[l]    = make_vec(n, 0.0f);
        net->err[l]    = make_vec(n, 0.0f);
    }

    srand(time(NULL));
    for(int l = 0; l < num_of_layers; l++) {
        int out = (l != num_of_layers - 1 ? hidden_size : output_size);
        int in  = (l == 0 ? input_size : hidden_size);
        for(int n = 0; n < out; n++) {
            for(int i = 0; i < in; i++) {
                net->weight[l]->dat[n][i] = (float)rand()*2.0f/RAND_MAX-1.0f;
            }
        }
    }

    return net;
}

void forward(Net *net, Vec *x, Vec *out) {
    for(int l = 0; l < net->num_of_layers; l++) {
        mat_vec_product(net->weight[l], (l == 0 ? x : net->act[l-1]), net->sum[l]);
        vec_sum(net->sum[l], net->bias[l]);

        if(l == net->num_of_layers - 1) continue;
        
        relu(net->sum[l], net->act[l]);

        printf("%d\n", l);
        dump_vec((l == 0 ? x : net->act[l-1]));
        dump_mat(net->weight[l]);
        dump_vec(net->sum[l]);
        dump_vec(net->act[l]);
    }
}

void free_net(Net *net) {
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