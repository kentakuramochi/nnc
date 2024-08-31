/**
 * @file nn_layer.c
 * @brief Layer structure
 *
 */
#include "nn_layer.h"

#include <stdbool.h>
#include <stdlib.h>

#define FREE_AND_NULL(ptr) { \
    free((ptr)); \
    (ptr) = NULL; \
}

NnLayer *nn_layer_alloc_params(NnLayer *layer) {
    if ((layer == NULL) ||
        // Layer type is NONE or not specified
        (layer->params.type == NN_LAYER_TYPE_NONE)) {
        return NULL;
    }

    return nn_layer_init_funcs[layer->params.type](layer);
}

void nn_layer_free_params(NnLayer *layer) {
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

void nn_layer_connect(NnLayer *prev, NnLayer *next) {
    next->params.batch_size = prev->params.batch_size;
    next->params.in = prev->params.out;
}

float *nn_layer_forward(NnLayer *layer, const float *x) {
    if ((layer == NULL) || (x == NULL)) {
        return NULL;
    }

    return layer->forward(layer, x);
}

float *nn_layer_backward(NnLayer *layer, const float *dy) {
    if ((layer == NULL) || (dy == NULL)) {
        return NULL;
    }

    return layer->backward(layer, dy);
}

void nn_layer_clear_grad(NnLayer *layer) {
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
