/**
 * @file scnn_layer.c
 * @brief Layer structure
 * 
 */
#include <stdlib.h>

#include "scnn_layer.h"

scnn_layer *scnn_layer_alloc(void)
{
    scnn_layer *layer = malloc(sizeof(scnn_layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->id = 0;

    layer->prev_id = 0;
    layer->next_id = 0;

    layer->forward  = NULL;
    layer->backward = NULL;

    return layer;
}

void scnn_layer_free(scnn_layer **layer)
{
    if ((layer == NULL) || (*layer == NULL)) {
        return;
    }

    scnn_layer *layer_ptr = *layer;

    free(layer_ptr->x.data);
    layer_ptr->x.data = NULL;
    free(layer_ptr->y.data);
    layer_ptr->y.data = NULL;
    free(layer_ptr->w.data);
    layer_ptr->w.data = NULL;
    free(layer_ptr->b.data);
    layer_ptr->b.data = NULL;

    free(layer_ptr->dx.data);
    layer_ptr->dx.data = NULL;
    free(layer_ptr->dw.data);
    layer_ptr->dw.data = NULL;
    free(layer_ptr->db.data);
    layer_ptr->db.data = NULL;

    free(*layer);
    *layer = NULL;
}
