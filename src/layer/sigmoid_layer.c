/**
 * @file sigmoid_layer.c
 * @brief Sigmoid layer
 *
 */
#include "layer/sigmoid_layer.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief Forward of the sigmoid layer
 *
 * @param[in,out] layer Layer
 * @param[in] x An input of the layer
 * @return Pointer to the layer output
 */
static float *sigmoid_forward(Layer *layer, const float *x) {
    LayerParams *params = &layer->params;

    for (int i = 0; i < (params->batch_size * params->in); i++) {
        layer->x[i] = x[i];
    }

    for (int i = 0; i < (params->batch_size * params->in); i++) {
        layer->y[i] = 1 / (1 + expf(-layer->x[i]));
    }

    return layer->y;
}

/**
 * @brief Backward of the sigmoid layer
 *
 * @param[in,out] layer Layer
 * @param[in] gy Gradient of the next layer
 * @return Pointer to gradient of the layer input
 */
static float *sigmoid_backward(Layer *layer, const float *gy) {
    LayerParams *params = &layer->params;

    for (int i = 0; i < (params->batch_size * params->in); i++) {
        layer->gx[i] = gy[i] * layer->y[i] * (1 - layer->y[i]);
    }

    return layer->gx;
}

Layer *sigmoid_layer_init(Layer *layer) {
    LayerParams *params = &layer->params;

    if ((params->batch_size == 0) ||
        (params->in == 0)) {
        return NULL;
    }

    size_t x_byte_size = sizeof(float) * params->in;
    layer->x = malloc(params->batch_size * x_byte_size);
    if (layer->x == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    layer->y = malloc(params->batch_size * x_byte_size);
    if (layer->y == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    layer->w = NULL;
    layer->b = NULL;

    layer->gx = malloc(params->batch_size * x_byte_size);
    if (layer->gx == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    // Set in = out if the allocation succeeded
    params->out = params->in;

    layer->gw = NULL;
    layer->gb = NULL;

    layer->forward = sigmoid_forward;
    layer->backward = sigmoid_backward;

    return layer;
}
