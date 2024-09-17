/**
 * @file softmax_layer.h
 * @brief Softmax layer
 *
 */
#include "layer/softmax_layer.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief Forward of the softmax layer
 *
 * @param[in,out] layer Layer
 * @param[in] x An input of the layer
 * @return Pointer to the layer output
 */
static float *softmax_forward(Layer *layer, const float *x) {
    LayerParams *params = &layer->params;

    for (int i = 0; i < (params->batch_size * params->in); i++) {
        layer->x[i] = x[i];
    }

    for (int i = 0; i < params->batch_size; i++) {
        int batch_idx = params->in * i;

        // Get a max. of the input to avoid overflow of exp(x)
        float c = -FLT_MAX;
        for (int j = 0; j < params->in; j++) {
            float a = layer->x[batch_idx + j];
            c = (a > c) ? a : c;
        }

        float sum = 0.0f;
        for (int j = 0; j < params->in; j++) {
            sum += expf(layer->x[batch_idx + j] - c);
        }

        for (int j = 0; j < params->in; j++) {
            layer->y[batch_idx + j] = expf(layer->x[batch_idx + j] - c) / sum;
        }
    }

    return layer->y;
}

/**
 * @brief Backward of the softmax layer
 *
 * @param[in,out] layer Layer
 * @param[in] gy Gradient of the next layer
 * @return Pointer to gradient of the layer input
 */
static float *softmax_backward(Layer *layer, const float *gy) {
    LayerParams *params = &layer->params;

    // Jacobian
    float *jacobian = malloc(params->in * params->out * sizeof(float));

    for (int i = 0; i < params->batch_size; i++) {
        int batch_idx = params->in * i;

        // Calculate a Jacobian
        for (int j = 0; j < params->out; j++) {
            for (int k = 0; k < params->in; k++) {
                if (j == k) {
                    jacobian[j * params->in + k] = layer->y[batch_idx + j] * (1 - layer->y[batch_idx + j]);
                } else {
                    jacobian[j * params->in + k] = -layer->y[batch_idx + j] * layer->y[batch_idx + k];
                }
            }
        }

        // Grad is calculated as a matrix multiplication of
        // the Jacobian and gradient of the next layer
        for (int j = 0; j < params->out; j++) {
            float acc = 0;
            for (int k = 0; k < params->in; k++) {
                acc += jacobian[j * params->in + k] * gy[batch_idx + k];
            }
            layer->gx[batch_idx + j] = acc;
        }
    }

    free(jacobian);

    return layer->gx;
}

Layer *softmax_layer_init(Layer *layer) {
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

    layer->forward = softmax_forward;
    layer->backward = softmax_backward;

    return layer;
}
