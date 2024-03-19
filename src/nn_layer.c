/**
 * @file nn_layer.c
 * @brief Layer structure
 *
 */
#include "nn_layer.h"

#include <stdbool.h>
#include <stdlib.h>

#include "activation.h"
#include "blas.h"

#define FREE_AND_NULL(ptr) { \
    free((ptr)); \
    (ptr) = NULL; \
}

float* forward(NnLayer *layer, const float *x) {
    const int m = layer->batch_size; // Batch dimension
    const int n = layer->out;
    const int k = layer->in;

    scopy((m * k), x, 1, layer->x, 1);

    // y = b: Broadcast for batch dimension
    for (int i = 0; i < m; i++) {
        scopy(n, layer->b, 1, &layer->y[i * n], 1);
    }

    // y = x * W + b
    sgemm(
        BLAS_NO_TRANS, BLAS_NO_TRANS,
        m, n, k,
        1.0, layer->x, k,
        layer->w, n,
        1.0, layer->y, n
    );

    // Activation
    sigmoid(layer->y, layer->z, (m * n));

    return layer->z;
}

float *backward(NnLayer *layer, const float *dy) {
    int m = layer->batch_size; // Batch dimension
    int n = layer->in;
    int k = layer->out;

    // dz = dy * z * (1 - z)
    for (int i = 0; i < (m * k); i++) {
        layer->dz[i] = dy[i] * layer->z[i] * (1.0f - layer->z[i]);
    }

    // dx = 0
    for (int i = 0; i < (m * n); i++) {
        layer->dx[i] = 0;
    }

    // dx = dy * WT
    sgemm(
        BLAS_NO_TRANS, BLAS_TRANS,
        m, n, k,
        1.0, layer->dz, k,
        layer->w, k,
        1.0, layer->dx, n
    );

    // dw = 0
    for (int i = 0; i < (n * k); i++) {
        layer->dw[i] = 0;
    }

    // dW = xT * dy
    sgemm(
        BLAS_TRANS, BLAS_NO_TRANS,
        n, k, m,
        1.0, layer->x, n,
        layer->dz, k,
        1.0, layer->dw, k
    );

    // db = 0
    for (int i = 0; i < k; i++) {
        layer->db[i] = 0;
    }

    // db = sum(dy) for batch
    for (int i = 0; i < m; i++) {
        saxpy(
            layer->out, 1, &layer->dz[i * k], 1,
            layer->db, 1
        );
    }

    return layer->dx;
}

NnLayer *nn_layer_alloc_params(NnLayer *layer) {
    if (layer == NULL) {
        return NULL;
    }

    if ((layer->batch_size == 0) ||
        (layer->in == 0) ||
        (layer->out == 0)) {
        return NULL;
    }

    size_t x_byte_size = sizeof(float) * layer->in;
    layer->x = malloc(layer->batch_size * x_byte_size);
    if (layer->x == NULL) {
        return NULL;
    }

    size_t y_byte_size = sizeof(float) * layer->out;
    layer->y = malloc(layer->batch_size * y_byte_size);
    if (layer->y == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->z = malloc(layer->batch_size * y_byte_size);
    if (layer->z == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    size_t w_byte_size = sizeof(float) * layer->in * layer->out;
    layer->w = malloc(w_byte_size);
    if (layer->w == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->b = malloc(y_byte_size);
    if (layer->b == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->dx = malloc(layer->batch_size * x_byte_size);
    if (layer->dx == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->dz = malloc(layer->batch_size * y_byte_size);
    if (layer->dz == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->dw = malloc(w_byte_size);
    if (layer->dw == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->db = malloc(y_byte_size);
    if (layer->db == NULL) {
        goto FREE_LAYER_PARAMS;
    }

    layer->forward = forward;
    layer->backward = backward;

    return layer;

FREE_LAYER_PARAMS:
    FREE_AND_NULL(layer->db);
    FREE_AND_NULL(layer->dw);
    FREE_AND_NULL(layer->dz);
    FREE_AND_NULL(layer->dx);
    FREE_AND_NULL(layer->b);
    FREE_AND_NULL(layer->w);
    FREE_AND_NULL(layer->z);
    FREE_AND_NULL(layer->y);
    FREE_AND_NULL(layer->x);

    return NULL;
}

void nn_layer_free_params(NnLayer *layer) {
    if (layer == NULL) {
        return;
    }

    FREE_AND_NULL(layer->x);
    FREE_AND_NULL(layer->y);
    FREE_AND_NULL(layer->z);
    FREE_AND_NULL(layer->w);
    FREE_AND_NULL(layer->b);
    FREE_AND_NULL(layer->dx);
    FREE_AND_NULL(layer->dz);
    FREE_AND_NULL(layer->dw);
    FREE_AND_NULL(layer->db);
}

void nn_layer_connect(NnLayer *prev, NnLayer *next) {
    next->batch_size = prev->batch_size;
    next->in = prev->out;
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

void nn_layer_update(NnLayer *layer, const float learning_rate) {
    const int w_size = layer->in * layer->out;

    saxpy(w_size, -learning_rate, layer->dw, 1, layer->w, 1);

    saxpy(layer->out, -learning_rate, layer->db, 1, layer->b, 1);
}
