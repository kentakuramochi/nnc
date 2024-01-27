/**
 * @file scnn_net.c
 * @brief Network structure
 * 
 */
#include "scnn_net.h"

#include <stdlib.h>
#include <stdbool.h>

int scnn_net_size(const scnn_net *net)
{
    return net->size;
}

int scnn_net_batch_size(const scnn_net *net)
{
    return net->batch_size;
}

scnn_layer *scnn_net_layers(scnn_net *net)
{
    return net->layers;
}

scnn_layer* scnn_net_input(const scnn_net *net)
{
    return net->input;
}

scnn_layer* scnn_net_output(const scnn_net *net)
{
    return net->output;
}

scnn_net *scnn_net_alloc(void)
{
    scnn_net *net = malloc(sizeof(scnn_net));
    if (net == NULL) {
        return NULL;
    }

    net->layers = NULL;

    net->size = 0;

    net->batch_size = 1;

    net->input  = NULL;
    net->output = NULL;

    return net;
}

scnn_net *scnn_net_append(scnn_net *net, scnn_layer_params params)
{
    if ((net == NULL) || (params.type == SCNN_LAYER_NONE)) {
        return NULL;
    }

    // Realloc layers when the number of layers exceed current allocated size
    scnn_layer *realloc_layers = realloc(net->layers, sizeof(scnn_layer) * (net->size + 1));
    if (realloc_layers == NULL) {
        return NULL;
    }

    net->layers = realloc_layers;
    realloc_layers = NULL;

    scnn_layer_connect(net->output, &net->layers[net->size]);

    // Set the first layer as a network input
    net->input = &net->layers[0];
    // And the last layer as a network output
    net->output = &net->layers[net->size];

    // Copy the layer parameters
    net->layers[net->size].params = params;

    net->size++;

    return net;
}

scnn_net *scnn_net_init(scnn_net *net)
{
    if ((net == NULL) || (net->size == 0)) {
        return NULL;
    }

    for (int i = 0; i < net->size; i++) {
        if (scnn_layer_init(&net->layers[i]) == NULL) {
            return NULL;
        }
    }

    return net;
}

#ifdef DEACTIVATE_TEMPORALLY
scnn_dtype *scnn_net_forward(scnn_net *net, const scnn_dtype *x)
{
    if ((net == NULL) || (x == NULL)) {
        return NULL;
    }

    scnn_dtype *in = (scnn_dtype*)x;
    scnn_dtype *out;
    for (int i = 0; i < net->size; i++) {
        out = scnn_layer_forward(net->layers[i], in);
        in = out;
    }

    return out;
}

scnn_dtype *scnn_net_backward(scnn_net *net, const scnn_dtype *dy)
{
    if ((net == NULL) || (dy == NULL)) {
        return NULL;
    }

    scnn_dtype *din = (scnn_dtype*)dy;
    scnn_dtype *dout;
    for (int i = (net->size - 1); i >= 0; i--) {
        dout = scnn_layer_backward(net->layers[i], din);
        din = dout;
    }

    return dout;
}
#endif

void scnn_net_free(scnn_net **net)
{
    if ((net == NULL) || (*net == NULL)) {
        return;
    }

    scnn_net *instance = *net;

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
