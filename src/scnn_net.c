/**
 * @file scnn_net.c
 * @brief Network structure
 * 
 */
#include <stdlib.h>
#include <stdbool.h>

#include "scnn_net.h"

scnn_net *scnn_net_alloc(void)
{
    scnn_net *net = malloc(sizeof(scnn_net));
    if (net == NULL) {
        return NULL;
    }

    // initialize member
    net->size = 0;

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
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

    if (net->size >= SCNN_NET_MAX_SIZE) {
        return NULL;
    }

    // check the matrix size
    if (net->output) {
        if ((net->output->y.n != layer->y.n) ||
            (net->output->y.c != layer->y.c) ||
            (net->output->y.h != layer->y.h) ||
            (net->output->y.w != layer->y.w)) {
            return NULL;
        }
    }

    layer->id      = net->size;
    layer->prev_id = -1;
    layer->next_id = -1;

    // attach layer id with the last layer
    scnn_layer *last_layer = net->output;
    if (last_layer != NULL) {
        last_layer->next_id = layer->id;
        layer->prev_id      = last_layer->id;
    }

    // append layer to tail of the network layers
    net->layers[net->size] = layer;

    if (net->size == 0) {
        // set the first layer as input
        net->input = layer;
    }

    // set the last layer as output
    net->output = layer;

    net->size++;

    return net;
}

//void scnn_net_forward(scnn_net *net, const scnn_mat *x)
//{
//}

//void scnn_net_backward(scnn_net *net, const scnn_mat *t)
//{
//}

void scnn_net_free(scnn_net **net)
{
    if (*net == NULL) {
        return;
    }

    scnn_layer *layer = (*net)->layers[0];
    if (layer == NULL) {
        goto NET_FREE;
    }

    for (int i = 0; i < (*net)->size; i++) {
        scnn_layer_free(&((*net)->layers[i]));
    }

NET_FREE:
    free(*net);
    *net = NULL;
}
