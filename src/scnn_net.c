/**
 * @file scnn_net.c
 * @brief Network structure
 * 
 */
#include <stdlib.h>
#include <stdbool.h>

#include "scnn_net.h"
#include "scnn_blas.h"

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
        if ((net->output->y.shape.d[0] != layer->y.shape.d[0]) ||
            (net->output->y.shape.d[1] != layer->y.shape.d[1]) ||
            (net->output->y.shape.d[2] != layer->y.shape.d[2]) ||
            (net->output->y.shape.d[3] != layer->y.shape.d[3])) {
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

void scnn_net_forward(scnn_net *net, const scnn_mat *x)
{
    if ((net == NULL) || (x == NULL)) {
        return;
    }

    scnn_mat *in = scnn_mat_alloc(x->shape);
    scnn_mat_copy_from_array(in, x->data, x->size);

    scnn_layer  *layer;
    for (int i = 0; i < net->size; i++) {
        layer = net->layers[i];
        layer->forward(layer, in);
        in = &(layer->y);
    }
}

void scnn_net_backward(scnn_net *net, const scnn_mat *t)
{
    if ((net == NULL) || (t == NULL)) {
        return;
    }

    scnn_mat *dy = scnn_mat_alloc(t->shape);

    scnn_scopy(net->output->y.size, net->output->y.data, 1, dy->data, 1);
    scnn_saxpy(t->size, -1, t->data, 1, dy->data, 1);

    scnn_mat    *out = dy;
    scnn_layer  *layer;
    for (int i = (net->size - 1); i >= 0; i--) {
        layer = net->layers[i];
        layer->backward(layer, out);
        out = &(layer->dx);
    }
}

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
