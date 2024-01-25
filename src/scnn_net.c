/**
 * @file scnn_net.c
 * @brief Network structure
 * 
 */
#include "scnn_net.h"

#include <stdlib.h>
#include <stdbool.h>

#define INIT_LAYER_SIZE 128 //!< The number of layers in network when initialization

/**
 * @brief Network structure
 * 
 */
struct scnn_net {
    int         size;       //!< The number of layers
    int         alloc_size; //!< Allocated size
    int         batch_size; //!< Batch size
    scnn_layer  **layers;    //!< Layers
    scnn_layer  *input;     //!< Input layer off the network
    scnn_layer  *output;    //!< Output layer off the network
};

int scnn_net_size(const scnn_net *net)
{
    return net->size;
}

int scnn_net_batch_size(const scnn_net *net)
{
    return net->batch_size;
}

scnn_layer **scnn_net_layers(scnn_net *net)
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

    net->layers = malloc(sizeof(scnn_net*) * INIT_LAYER_SIZE);
    if (net->layers == NULL) {
        return NULL;
    }

    net->size = 0;
    net->alloc_size = INIT_LAYER_SIZE;

    net->batch_size = 1;

    for (int i = 0; i < INIT_LAYER_SIZE; i++) {
        net->layers[i] = NULL;
    }

    net->input  = NULL;
    net->output = NULL;

    return net;
}

scnn_net *scnn_net_append(scnn_net *net, scnn_layer *layer)
{
    if ((net == NULL) || (layer == NULL)) {
        return NULL;
    }

    // Realloc layers when the number of layers exceed current allocated size
    if (net->size == net->alloc_size) {
        int realloc_size = net->size + INIT_LAYER_SIZE;

        scnn_layer **realloc_layers = realloc(net->layers, sizeof(scnn_layer*) * realloc_size);
        if (realloc_layers == NULL) {
            scnn_net_free(&net);
            return NULL;
        }

        net->layers = realloc_layers;
    }

    scnn_layer_connect(net->output, layer);

    // Append the current layer to tail of the network layers
    net->layers[net->size] = layer;

    // Set the first layer as a network input
    if (net->size == 0) {
        net->input = layer;
    }
    // And the last layer as a network output
    net->output = layer;

    net->size++;

    return net;
}

scnn_net *scnn_net_init(scnn_net *net)
{
    if ((net == NULL) || (net->size == 0)) {
        return NULL;
    }

    for (int i = 0; i < net->size; i++) {
        if (scnn_layer_init(net->layers[i]) == NULL) {
            return NULL;
        }
    }

    return net;
}

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

void scnn_net_free(scnn_net **net)
{
    if ((net == NULL) || (*net == NULL)) {
        return;
    }

    for (int i = 0; i < (*net)->size; i++) {
        scnn_layer_free(&((*net)->layers[i]));
    }

    free((*net)->layers);
    (*net)->layers = NULL;

    free(*net);
    *net = NULL;
}
