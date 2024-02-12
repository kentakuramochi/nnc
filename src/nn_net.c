/**
 * @file nn_net.c
 * @brief Network structure
 *
 */
#include "nn_net.h"

#include <stdlib.h>
#include <stdbool.h>

int nn_net_size(const NnNet *net) {
    return net->size;
}

int nn_net_batch_size(const NnNet *net) {
    return net->batch_size;
}

NnLayer *nn_net_layers(NnNet *net) {
    return net->layers;
}

NnLayer *nn_net_input(const NnNet *net) {
    return &net->layers[0];
}
 
NnLayer *nn_net_output(const NnNet *net) {
    if (net->size == 0) {
        return NULL;
    }
    return &net->layers[net->size - 1];
}

NnNet *nn_net_alloc(void) {
    NnNet *net = malloc(sizeof(NnNet));
    if (net == NULL) {
        return NULL;
    }

    net->size = 0;
    net->batch_size = 1;
    net->layers = NULL;
    // net->input = NULL;
    // net->output = NULL;

    return net;
}

NnNet *nn_net_append(NnNet *net, NnLayerParams params) {
    if (net == NULL) {
        return NULL;
    }

    // Reallocate and extend layers in the network
    NnLayer *realloc_layers = realloc(net->layers, sizeof(NnLayer) * (net->size + 1));
    if (realloc_layers == NULL) {
        return NULL;
    }

    net->layers = realloc_layers;
    realloc_layers = NULL;

    // Initialize new layer
    NnLayer *layer = &net->layers[net->size];
    layer->in = params.in;
    layer->out = params.out;
    layer->x = NULL;
    layer->y = NULL;
    layer->w = NULL;
    layer->b = NULL;
    layer->dx = NULL;
    layer->dw = NULL;
    layer->db = NULL;

    net->size++;

    // Connect the layer
    for (int i = 1; i < net->size; i++) {
        nn_layer_connect(&net->layers[i - 1], &net->layers[i]);
    }

    return net;
}

NnNet *nn_net_init(NnNet *net) {
    if ((net == NULL) || (net->size == 0)) {
        return NULL;
    }

    for (int i = 0; i < net->size; i++) {
        if (nn_layer_init(&net->layers[i]) == NULL) {
            return NULL;
        }
    }

    return net;
}

float *nn_net_forward(NnNet *net, const float *x) {
    if ((net == NULL) || (x == NULL)) {
        return NULL;
    }

    float *in = (float*)x;
    float *out;
    for (int i = 0; i < net->size; i++) {
        out = nn_layer_forward(&net->layers[i], in);
        in = out;
    }

    return out;
}

float *nn_net_backward(NnNet *net, const float *dy) {
    if ((net == NULL) || (dy == NULL)) {
        return NULL;
    }

    float *din = (float*)dy;
    float *dout;
    for (int i = (net->size - 1); i >= 0; i--) {
        dout = nn_layer_backward(&net->layers[i], din);
        din = dout;
    }

    return dout;
}

void nn_net_update(NnNet *net, const float learning_rate) {
    for (int i = 0; i < net->size; i++) {
        nn_layer_update(&net->layers[i], learning_rate);
    }
}

void nn_net_free(NnNet **net) {
    if ((net == NULL) || (*net == NULL)) {
        return;
    }

    NnNet *instance = *net;

    for (int i = 0; i < instance->size; i++) {
        free(instance->layers[i].x);
        instance->layers[i].x = NULL;
        free(instance->layers[i].y);
        instance->layers[i].y = NULL;
        free(instance->layers[i].w);
        instance->layers[i].w = NULL;
        free(instance->layers[i].b);
        instance->layers[i].b = NULL;
        free(instance->layers[i].dx);
        instance->layers[i].dx = NULL;
        free(instance->layers[i].dw);
        instance->layers[i].dw = NULL;
        free(instance->layers[i].db);
        instance->layers[i].db = NULL;
    }

    free(instance->layers);
    instance->layers = NULL;

    free(*net);
    *net = NULL;
}
