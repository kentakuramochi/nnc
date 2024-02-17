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

NnLayer *nn_layer_alloc(const NnLayerParams params) {
    NnLayer *layer = malloc(sizeof(NnLayer));
    if (layer == NULL) {
        return NULL;
    }

    layer->batch_size = params.batch_size;
    layer->in = params.in;
    layer->out = params.out;

    layer->x = NULL;
    layer->y = NULL;
    layer->z = NULL;
    layer->w = NULL;
    layer->b = NULL;

    layer->dx = NULL;
    layer->dz = NULL;
    layer->dw = NULL;
    layer->db = NULL;

    return layer;
}

NnLayer *nn_layer_init(NnLayer *layer) {
    if (layer == NULL) {
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
        goto FREE_MATRICES;
    }

    layer->z = malloc(layer->batch_size * y_byte_size);
    if (layer->z == NULL) {
        goto FREE_MATRICES;
    }

    size_t w_byte_size = sizeof(float) * layer->in * layer->out;
    layer->w = malloc(w_byte_size);
    if (layer->w == NULL) {
        goto FREE_MATRICES;
    }

    layer->b = malloc(y_byte_size);
    if (layer->b == NULL) {
        goto FREE_MATRICES;
    }

    layer->dx = malloc(layer->batch_size * x_byte_size);
    if (layer->dx == NULL) {
        goto FREE_MATRICES;
    }

    layer->dz = malloc(layer->batch_size * y_byte_size);
    if (layer->dz == NULL) {
        goto FREE_MATRICES;
    }

    layer->dw = malloc(w_byte_size);
    if (layer->dw == NULL) {
        goto FREE_MATRICES;
    }

    layer->db = malloc(y_byte_size);
    if (layer->db == NULL) {
        goto FREE_MATRICES;
    }

    return layer;

FREE_MATRICES:
    free(layer->db);
    layer->db = NULL;
    free(layer->dw);
    layer->dw = NULL;
    free(layer->dz);
    layer->dz = NULL;
    free(layer->dx);
    layer->dx = NULL;
    free(layer->b);
    layer->b = NULL;
    free(layer->w);
    layer->w = NULL;
    free(layer->z);
    layer->z = NULL;
    free(layer->y);
    layer->y = NULL;
    free(layer->x);
    layer->x = NULL;

    return NULL;
}

void nn_layer_connect(NnLayer *prev, NnLayer *next) {
    next->batch_size = prev->batch_size;
    next->in = prev->out;
}

float *nn_layer_forward(NnLayer *layer, const float *x) {
    if ((layer == NULL) || (x == NULL)) {
        return NULL;
    }

    const int m = layer->batch_size; // Batch dimension
    const int n = layer->out;
    const int k = layer->in;

    for (int i = 0; i < m; i++) {
        scopy(
            layer->in, x, 1, &layer->x[i * layer->in], 1
        );
    }

    // y = b: Broadcast for batch dimension
    for (int i = 0; i < m; i++) {
        scopy(
            layer->out, layer->b, 1, &layer->y[i * layer->out], 1
        );
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
    sigmoid(layer->y, layer->z, (layer->batch_size * layer->out));

    return layer->z;
}

float *nn_layer_backward(NnLayer *layer, const float *dy) {
    if ((layer == NULL) || (dy == NULL)) {
        return NULL;
    }

    int m = layer->batch_size; // Batch dimension
    int n = layer->in;
    int k = layer->out;

    // dz = dy * z * (1 - z)
    for (int i = 0; i < (m * layer->out); i++) {
        layer->dz[i] = dy[i] * layer->z[i] * (1.0f - layer->z[i]);
    }

    // dx = 0
    for (int i = 0; i < (m * layer->in); i++) {
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
    for (int i = 0; i < layer->in * layer->out; i++) {
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
    for (int i = 0; i < layer->out; i++) {
        layer->db[i] = 0;
    }

    // db = sum(dy) for batch
    for (int i = 0; i < layer->batch_size; i++) {
        saxpy(
            layer->out, 1, &layer->dz[i * layer->out], 1,
            layer->db, 1
        );
    }

    return layer->dx;
}

void nn_layer_update(NnLayer *layer, const float learning_rate) {
    const int w_size = layer->in * layer->out;

    saxpy(w_size, -learning_rate, layer->dw, 1, layer->w, 1);

    saxpy(layer->out, -learning_rate, layer->db, 1, layer->b, 1);
}

void nn_layer_free(NnLayer **layer) {
    if ((layer == NULL) || (*layer == NULL)) {
        return;
    }

    free((*layer)->x);
    (*layer)->x = NULL;
    free((*layer)->y);
    (*layer)->y = NULL;
    free((*layer)->z);
    (*layer)->z = NULL;
    free((*layer)->w);
    (*layer)->w = NULL;
    free((*layer)->b);
    (*layer)->b = NULL;

    free((*layer)->dx);
    (*layer)->dx = NULL;
    free((*layer)->dz);
    (*layer)->dz = NULL;
    free((*layer)->dw);
    (*layer)->dw = NULL;
    free((*layer)->db);
    (*layer)->db = NULL;

    free(*layer);
    *layer = NULL;
}
