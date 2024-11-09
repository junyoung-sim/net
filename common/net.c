#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "net.h"

float rand_normal() {
    float r1 = (float)(rand() + 0.5f) / (RAND_MAX + 1.0f);
    float r2 = (float)(rand() + 0.5f) / (RAND_MAX + 1.0f);
    return sqrt(-2.0f * log(r1)) * sin(2.0f * PI * r2);
}

void linear(Vec *x, Vec *out) {
    for(int i = 0; i < x->size; i++) {
        out->dat[i] = x->dat[i];
    }
}

void relu(Vec *x, Vec *out) {
    for(int i = 0; i < x->size; i++) {
        out->dat[i] = (x->dat[i] < 0.0f ? 0.0f : x->dat[i]);
    }
}

void sigmoid(Vec *x, Vec *out) {
    for(int i = 0; i < x->size; i++) {
        out->dat[i] = 1.0f / (1.0f + exp(-x->dat[i]));
    }
}

void softmax(Vec *x, Vec *out) {
    float norm = 0.0f;
    for(int i = 0; i < x->size; i++) {
        norm += exp(x->dat[i]);
    }
    for(int i = 0; i < x->size; i++) {
        out->dat[i] = exp(x->dat[i]) / norm;
    }
}

Net *make_net(int* shape, int num_of_layers, int input_size, int output_type) {
    Net *net = calloc(1, sizeof(Net));
    net->shape         = shape;
    net->input_size    = input_size;
    net->output_type   = output_type;
    net->num_of_layers = num_of_layers;

    net->grad   = calloc(num_of_layers, sizeof(Mat*));
    net->weight = calloc(num_of_layers, sizeof(Mat*));
    net->bias   = calloc(num_of_layers, sizeof(Vec*));
    net->sum    = calloc(num_of_layers, sizeof(Vec*));
    net->act    = calloc(num_of_layers, sizeof(Vec*));
    net->err    = calloc(num_of_layers, sizeof(Vec*));

    for(int l = 0; l < num_of_layers; l++) {
        int out = shape[l];
        int in  = (l == 0 ? input_size : shape[l-1]);

        net->grad[l]   = make_mat(out, in+1, 0.0f);
        net->weight[l] = make_mat(out, in, 0.0f);
        net->bias[l]   = make_vec(out, 0.0f);
        net->sum[l]    = make_vec(out, 0.0f);
        net->act[l]    = make_vec(out, 0.0f);
        net->err[l]    = make_vec(out, 0.0f);
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

Vec* forward(Net *net, Vec *x) {
    for(int l = 0; l < net->num_of_layers; l++) {
        mat_vec_product(net->weight[l], (l == 0 ? x : net->act[l-1]), net->sum[l]);
        vec_sum(net->sum[l], net->bias[l]);

        if(l == net->num_of_layers - 1) continue;
        
        relu(net->sum[l], net->act[l]);
    }

    int lout = net->num_of_layers - 1;
    switch(net->output_type) {
        case LINEAR:
            linear(net->sum[lout], net->act[lout]);
            break;
        case SIGMOID:
            sigmoid(net->sum[lout], net->act[lout]);
            break;
        case SOFTMAX:
            softmax(net->sum[lout], net->act[lout]);
            break;
        default:
            linear(net->sum[lout], net->act[lout]);
            break;
    }

    return net->act[lout];
}

void backward(Net *net, Vec *x, Vec *y, float alpha, float lambda) {
}

/*void MLP::backward(std::vector<float> &x, std::vector<float> &y, float alpha, float lambda) {
    assert(initialized && x.size() == _input_size && y.size() == _shape.back());
    std::vector<float> out = forward(x);
    //std::vector<float> ierr(_input_size, 0.0f);
    for(int l = _shape.size() - 1; l >= 0; l--) {
        float partial_gradient = 0.0f, full_gradient = 0.0f;
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            if(l == _shape.size() - 1) {
                if(_output_type == "linear") partial_gradient = -2.0f * (y[n] - out[n]);
                if(_output_type == "sigmoid" || _output_type == "softmax") partial_gradient = out[n] - y[n];
            }
            else partial_gradient = _err[l][n] * drelu(_sum[l][n]);

            _bias_grad[l][n] += alpha * partial_gradient;
            for(unsigned int i = 0; i < in_features; i++) {
                if(l == 0) {
                    full_gradient = partial_gradient * x[i];
                    //ierr[i] += partial_gradient * _weight[l][n][i];
                }
                else {
                    full_gradient = partial_gradient * _act[l-1][i];
                    _err[l-1][i] += partial_gradient * _weight[l][n][i];
                }
                full_gradient += lambda * _weight[l][n][i];
                _weight_grad[l][n][i] += alpha * full_gradient;
            }
        }
    }
    _backward_count++;
    //return ierr;
}

void MLP::step() {
    assert(initialized && _backward_count != 0);
    for(unsigned int l = 0; l < _shape.size(); l++) {
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            _bias[l][n] -= _bias_grad[l][n] / _backward_count;
            for(unsigned int i = 0; i < in_features; i++)
                _weight[l][n][i] -= _weight_grad[l][n][i] / _backward_count;
        }
    }
}

void MLP::zero_grad() {
    assert(initialized);
    for(unsigned int l = 0; l < _shape.size(); l++) {
        unsigned int in_features = (l == 0 ? _input_size : _shape[l-1]);
        for(unsigned int n = 0; n < _shape[l]; n++) {
            _bias_grad[l][n] = 0.0f;
            for(unsigned int i = 0; i < in_features; i++)
                _weight_grad[l][n][i] = 0.0f;
        }
    }
    _backward_count = 0;
}*/

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