/**
 * @file scnn_layer.c
 * @brief Layer structure
 * 
 */
#include "scnn_layer.h"

#include <stdlib.h>
#include <stdbool.h>

#include "scnn_fc.h"

scnn_dtype *scnn_layer_y(const scnn_layer *layer)
{
    // TODO: Implement
    return NULL;
}

scnn_layer *scnn_layer_alloc(const scnn_layer_params params)
{
    scnn_layer *layer = malloc(sizeof(scnn_layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->params = params;

    layer->x = NULL;
    layer->y = NULL;
    layer->w = NULL;
    layer->b = NULL;

    layer->dx = NULL;
    layer->dw = NULL;
    layer->db = NULL;

    layer->init = NULL;

    layer->forward  = NULL;
    layer->backward = NULL;

    return layer;
}

scnn_layer *scnn_layer_init(scnn_layer* layer)
{
    if (layer == NULL) {
        return NULL;
    }

    size_t x_size = sizeof(scnn_dtype)
        * layer->params.in_shape[0]
        * layer->params.in_shape[1]
        * layer->params.in_shape[2]
        * layer->params.in_shape[3];

    layer->x = malloc(x_size);
    if (layer->x == NULL) {
        return NULL;
    }

    size_t y_size = sizeof(scnn_dtype) * layer->params.out;
    layer->y = malloc(y_size);
    if (layer->y == NULL) {
        goto FREE_MATRICES;
    }

    size_t w_size = x_size * y_size;
    layer->w = malloc(w_size);
    if (layer->w == NULL) {
        goto FREE_MATRICES;
    }

    layer->b = malloc(y_size);
    if (layer->b == NULL) {
        goto FREE_MATRICES;
    }

    layer->dx = malloc(x_size);
    if (layer->dx == NULL) {
        goto FREE_MATRICES;
    }

    layer->dw = malloc(w_size);
    if (layer->dw == NULL) {
        goto FREE_MATRICES;
    }

    layer->db = malloc(y_size);
    if (layer->db == NULL) {
        goto FREE_MATRICES;
    }

    return layer;

FREE_MATRICES:
    free(layer->db);
    layer->db = NULL;
    free(layer->dw);
    layer->dw = NULL;
    free(layer->dx);
    layer->dx = NULL;
    free(layer->b);
    layer->b = NULL;
    free(layer->w);
    layer->w = NULL;
    free(layer->y);
    layer->y = NULL;
    free(layer->x);
    layer->x = NULL;

    return NULL;
}

void scnn_layer_connect(scnn_layer* prev, scnn_layer* next)
{
    next->params.in_shape[0] = 1;
    next->params.in_shape[1] = prev->params.out;
    next->params.in_shape[2] = 1;
    next->params.in_shape[3] = 1;

    return;
}

scnn_dtype *scnn_layer_forward(scnn_layer *layer, const scnn_dtype *x)
{
    if ((layer == NULL) || (x == NULL)) {
        return NULL;
    }

    return scnn_fc(x, layer->w, layer->b, layer->y);
}

scnn_dtype *scnn_layer_backward(scnn_layer *layer, const scnn_dtype *dy)
{
    if ((layer == NULL) || (dy == NULL)) {
        return NULL;
    }

    return scnn_fc_diff(dy, layer->w, layer->b, layer->dx, layer->dw, layer->db);
}

void scnn_layer_free(scnn_layer **layer)
{
    if ((layer == NULL) || (*layer == NULL)) {
        return;
    }

    free((*layer)->x);
    (*layer)->x = NULL;
    free((*layer)->y);
    (*layer)->y = NULL;
    free((*layer)->w);
    (*layer)->w = NULL;
    free((*layer)->b);
    (*layer)->b = NULL;

    free((*layer)->dx);
    (*layer)->dx = NULL;
    free((*layer)->dw);
    (*layer)->dw = NULL;
    free((*layer)->db);
    (*layer)->db = NULL;

    free(*layer);
    *layer = NULL;
}
