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
static float *sigmoid_forward(NnLayer *layer, const float *x) {
    NnLayerParams *params = &layer->params;

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
 * @param[in] dy A differential of previous layer
 * @return Pointer to differential of an input of the layer
 */
static float *sigmoid_backward(NnLayer *layer, const float *dy) {
    NnLayerParams *params = &layer->params;

    for (int i = 0; i < (params->batch_size * params->in); i++) {
        layer->dx[i] = dy[i] * layer->y[i] * (1 - layer->y[i]);
    }

    return layer->dx;
}

NnLayer *sigmoid_layer_init(NnLayer *layer) {
    NnLayerParams *params = &layer->params;

    if ((params->batch_size == 0) ||
        (params->in == 0)) {
        return NULL;
    }

    size_t x_byte_size = sizeof(float) * params->in;
    layer->x = malloc(params->batch_size * x_byte_size);
    if (layer->x == NULL) {
        nn_layer_free_params(layer);
        return NULL;
    }

    layer->y = malloc(params->batch_size * x_byte_size);
    if (layer->y == NULL) {
        nn_layer_free_params(layer);
        return NULL;
    }

    layer->w = NULL;
    layer->b = NULL;

    layer->dx = malloc(params->batch_size * x_byte_size);
    if (layer->dx == NULL) {
        nn_layer_free_params(layer);
        return NULL;
    }

    layer->dw = NULL;
    layer->db = NULL;

    layer->forward = sigmoid_forward;
    layer->backward = sigmoid_backward;

    return layer;
}
