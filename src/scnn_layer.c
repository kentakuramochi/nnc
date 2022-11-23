/**
 * @file scnn_layer.c
 * @brief Layer structure
 * 
 */
#include <stdlib.h>
#include <stdbool.h>

#include "scnn_layer.h"

scnn_layer *scnn_layer_alloc(const scnn_layer_params params)
{
    scnn_layer *layer = malloc(sizeof(scnn_layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->id       = 0;
    layer->prev_id  = 0;
    layer->next_id  = 0;

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
