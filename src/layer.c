/**
 * @file layer.c
 * @brief Layer structure
 *
 */
#include "layer.h"

#include <stdbool.h>
#include <stdlib.h>

#define FREE_AND_NULL(ptr) { \
    free((ptr)); \
    (ptr) = NULL; \
}

Layer *layer_alloc_params(Layer *layer) {
    if ((layer == NULL) ||
        // Layer type is NONE or not specified
        (layer->params.type == LAYER_TYPE_NONE)) {
        return NULL;
    }

    return layer_init_funcs[layer->params.type](layer);
}

void layer_free_params(Layer *layer) {
    if (layer == NULL) {
        return;
    }

    FREE_AND_NULL(layer->x);
    FREE_AND_NULL(layer->y);
    FREE_AND_NULL(layer->w);
    FREE_AND_NULL(layer->b);
    FREE_AND_NULL(layer->dx);
    FREE_AND_NULL(layer->dw);
    FREE_AND_NULL(layer->db);

    layer->forward = NULL;
    layer->backward = NULL;
}

bool layer_connect(Layer *prev, Layer *next) {
    LayerParams *n_params = &next->params;
    if ((n_params->batch_size != 0) ||
        (n_params->in !=0)) {
        return false;
    }

    n_params->batch_size = prev->params.batch_size;
    n_params->in = prev->params.out;

    return true;
}

float *layer_forward(Layer *layer, const float *x) {
    if ((layer == NULL) || (x == NULL)) {
        return NULL;
    }

    return layer->forward(layer, x);
}

float *layer_backward(Layer *layer, const float *dy) {
    if ((layer == NULL) || (dy == NULL)) {
        return NULL;
    }

    return layer->backward(layer, dy);
}

void layer_clear_grad(Layer *layer) {
    if (layer->dx != NULL) {
        const int x_size = layer->params.batch_size * layer->params.in;
        for (int i = 0; i < x_size; i++) {
            layer->dx[i] = 0;
        }
    }

    if (layer->dw != NULL) {
        const int w_size = layer->params.in * layer->params.out;
        for (int i = 0; i < w_size; i++) {
            layer->dw[i] = 0;
        }
    }

    if (layer->db != NULL) {
        for (int i = 0; i < layer->params.out; i++) {
            layer->db[i] = 0;
        }
    }
}
