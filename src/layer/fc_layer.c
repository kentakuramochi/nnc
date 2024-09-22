/**
 * @file fc_layer.c
 * @brief Fully connected layer
 */
#include "layer/fc_layer.h"

#include <stdlib.h>

/**
 * @brief Forward of the FC layer
 *
 * @param[in,out] layer Layer
 * @param[in] x An input of the layer
 * @return Pointer to the layer output
 */
static float *fc_forward(Layer *layer, const float *x) {
    LayerParams *params = &layer->params;

    for (int i = 0; i < (params->batch_size * params->in); i++) {
        layer->x[i] = x[i];
    }

    for (int i = 0; i < params->batch_size; i++) {
        for (int j = 0; j < params->out; j++) {
            float mac = 0;

            for (int k = 0; k < params->in; k++) {
                mac += layer->w[j * params->in + k] * layer->x[i * params->in + k];
            }
            mac += layer->b[j];

            layer->y[i * params->out + j] = mac;
        }
    }

    return layer->y;
}

/**
 * @brief Backward of the FC layer
 *
 * @param[in,out] layer Layer
 * @param[in] gy Gradient of the next layer
 * @return Pointer to gradient of the layer input
 */
static float *fc_backward(Layer *layer, const float *gy) {
    LayerParams *params = &layer->params;

    for (int i = 0; i < params->batch_size; i++) {
        for (int j = 0; j < params->in; j++) {
            float mac = 0;

            for (int k = 0; k < params->out; k++) {
                mac += gy[i * params->out + k] * layer->w[k * params->in + j];
            }

            layer->gx[i * params->in + j] += mac;
        }
    }

    for (int i = 0; i < params->in; i++) {
        for (int j = 0; j < params->out; j++) {
            float mac = 0;

            for (int k = 0; k < params->batch_size; k++) {
                mac += layer->x[k * params->in + i] * gy[k * params->out + j];
            }

            layer->gw[j * params->in + i] += mac;
        }
    }

    for (int i = 0; i < params->out; i++) {
        for (int j = 0; j < params->batch_size; j++) {
            layer->gb[i] += gy[j * params->out + i];
        }
    }

    return layer->gx;
}

Layer *fc_layer_init(Layer *layer) {
    LayerParams *params = &layer->params;

    if ((params->batch_size == 0) ||
        (params->in == 0) ||
        (params->out == 0)) {
        return NULL;
    }

    size_t x_byte_size = sizeof(float) * params->in;
    layer->x = malloc(params->batch_size * x_byte_size);
    if (layer->x == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    size_t y_byte_size = sizeof(float) * params->out;
    layer->y = malloc(params->batch_size * y_byte_size);
    if (layer->y == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    size_t w_byte_size = sizeof(float) * params->in * params->out;
    layer->w = malloc(params->batch_size * w_byte_size);
    if (layer->w == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    layer->b = malloc(params->batch_size * y_byte_size);
    if (layer->b == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    layer->gx = malloc(params->batch_size * x_byte_size);
    if (layer->gx == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    layer->gw = malloc(params->batch_size * w_byte_size);
    if (layer->gw == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    layer->gb = malloc(params->batch_size * y_byte_size);
    if (layer->gb == NULL) {
        layer_free_params(layer);
        return NULL;
    }

    layer->forward = fc_forward;
    layer->backward = fc_backward;

    return layer;
}
