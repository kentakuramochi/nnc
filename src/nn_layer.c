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
    const int x_size = layer->params.batch_size * layer->params.in;
    for (int i = 0; i < x_size; i++) {
        layer->dx[i] = 0;
    }

    const int w_size = layer->params.in * layer->params.out;
    for (int i = 0; i < w_size; i++) {
        layer->dw[i] = 0;
    }

    for (int i = 0; i < layer->params.out; i++) {
        layer->db[i] = 0;
    }
}

NnLayer *identity_init(NnLayer *layer) {
    NnLayerParams *params = &layer->params;

    if ((params->batch_size == 0) ||
        (params->in == 0) ||
        (params->out == 0)) {
        return NULL;
    }

    size_t x_byte_size = sizeof(float) * params->in;
    layer->x = malloc(params->batch_size * x_byte_size);
    if (layer->x == NULL) {
        return NULL;
    }

    size_t y_byte_size = sizeof(float) * params->out;
    layer->y = malloc(params->batch_size * y_byte_size);
    if (layer->y == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->dx = malloc(params->batch_size * x_byte_size);
    if (layer->dx == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->forward = identity_forward;
    layer->backward = identity_backward;

    return layer;

FREE_LAYER_PARAMS:
    nn_layer_free_params(layer);

    return NULL;
}

float *identity_forward(NnLayer *layer, const float *x) {
    // Just a reference
    for (int i = 0; i < layer->params.in; i++) {
        layer->y[i] = layer->x[i];
    }

    return layer->y;
}

float *identity_backward(NnLayer *layer, const float *dy) {
    // Just a reference
    for (int i = 0; i < layer->params.out; i++) {
        layer->y[i] = layer->dx[i];
    }

    return layer->dx;
}
