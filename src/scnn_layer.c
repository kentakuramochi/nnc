/**
 * @file scnn_layer.c
 * @brief Layer structure
 * 
 */
#include "scnn_layer.h"

#include <stdlib.h>
#include <stdbool.h>

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

    layer->params.id       = 0;
    layer->params.prev_id  = 0;
    layer->params.next_id  = 0;

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
    if ((layer == NULL) || (layer->init == NULL)) {
        return NULL;
    }
    return layer->init(layer);
}

void scnn_layer_connect(scnn_layer* prev, scnn_layer* next)
{
    prev->params.next_id = next->params.id;

    next->params.prev_id = prev->params.id;

    return;
}

scnn_dtype *scnn_layer_forward(scnn_layer *layer, const scnn_dtype *x)
{
    if ((layer == NULL) || (x == NULL) || (layer->forward == NULL)) {
        return NULL;
    }
    return layer->forward(layer, x);
}

scnn_dtype *scnn_layer_backward(scnn_layer *layer, const scnn_dtype *dy)
{
    if ((layer == NULL) || (dy == NULL) || (layer->backward == NULL)) {
        return NULL;
    }
    return layer->backward(layer, dy);
}

void scnn_layer_free(scnn_layer **layer)
{
    if ((layer == NULL) || (*layer == NULL)) {
        return;
    }

    scnn_mat_free(&(*layer)->x);
    scnn_mat_free(&(*layer)->y);
    scnn_mat_free(&(*layer)->w);
    scnn_mat_free(&(*layer)->b);

    scnn_mat_free(&(*layer)->dx);
    scnn_mat_free(&(*layer)->dw);
    scnn_mat_free(&(*layer)->db);

    free(*layer);
    *layer = NULL;
}
