/**
 * @file scnn_layer.c
 * @brief Layer structure
 * 
 */
#include <stdlib.h>

#include "scnn_layer.h"

scnn_layer *scnn_layer_alloc(const scnn_layer_params params)
{
    scnn_layer *layer = malloc(sizeof(scnn_layer));
    if (layer == NULL) {
        return NULL;
    }

    // count num of dimension
    int n_dim = 0;
    for (int i = 0; i < 4; i++) {
        if (params.in_shape[i] > 0) {
            n_dim++;
        }
    }
    // set 4-d shape with considering with omitted dimension
    // omitted dimenstion is set to 1
    int shape_idx = n_dim - 4;
    for (int i = 0; i < 4; i++) {
        layer->params.in_shape[i] = ((shape_idx >= 0) ? params.in_shape[shape_idx] : 1);
        shape_idx++;
    }

    layer->x.data = NULL;
    layer->y.data = NULL;
    layer->w.data = NULL;
    layer->b.data = NULL;

    layer->dx.data = NULL;
    layer->dw.data = NULL;
    layer->db.data = NULL;

    layer->id = 0;

    layer->prev_id = 0;
    layer->next_id = 0;

    layer->forward  = NULL;
    layer->backward = NULL;

    layer->set_size = NULL;

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
