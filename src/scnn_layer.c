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

    // count num of dimension
    int  n_dim = 0;
    bool has_dim_zero = false;
    for (int i = 0; i < 4; i++) {
        if (params.in_shape[i] > 0) {
            if (has_dim_zero) {
                return NULL;
            }
            n_dim++;
        } else if (params.in_shape[i] < 0) {
            return NULL;
        } else { // zero
            has_dim_zero = true;
        }
    }
    // set 4-d shape with considering with omitted dimension
    // omitted dimenstion is set to 1
    int shape_idx = n_dim - 4;
    for (int i = 0; i < 4; i++) {
        layer->params.in_shape[i] = ((shape_idx >= 0) ? params.in_shape[shape_idx] : 1);
        shape_idx++;
    }

    layer->x = NULL;
    layer->y = NULL;
    layer->w = NULL;
    layer->b = NULL;

    layer->dx = NULL;
    layer->dw = NULL;
    layer->db = NULL;

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

    scnn_mat_free(&layer_ptr->x);
    scnn_mat_free(&layer_ptr->y);
    scnn_mat_free(&layer_ptr->w);
    scnn_mat_free(&layer_ptr->b);

    scnn_mat_free(&layer_ptr->dx);
    scnn_mat_free(&layer_ptr->dw);
    scnn_mat_free(&layer_ptr->db);

    free(*layer);
    *layer = NULL;
}
