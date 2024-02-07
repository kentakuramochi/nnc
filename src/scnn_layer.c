/**
 * @file scnn_layer.c
 * @brief Layer structure
 * 
 */
#include "scnn_layer.h"

#include <stdbool.h>
#include <stdlib.h>

#include "activation.h"
#include "scnn_blas.h"

scnn_layer *scnn_layer_alloc(const scnn_layer_params params)
{
    scnn_layer *layer = malloc(sizeof(scnn_layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->params = params;

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

scnn_layer *scnn_layer_init(scnn_layer* layer)
{
    if (layer == NULL) {
        return NULL;
    }

    size_t x_size = sizeof(float) * layer->params.in;
    layer->x = malloc(x_size);
    if (layer->x == NULL) {
        return NULL;
    }

    size_t y_size = sizeof(float) * layer->params.out;
    layer->y = malloc(y_size);
    if (layer->y == NULL) {
        goto FREE_MATRICES;
    }

    layer->z = malloc(y_size);
    if (layer->z == NULL) {
        goto FREE_MATRICES;
    }

    size_t w_size = x_size * y_size;
    layer->w = malloc(w_size);
    if (layer->w == NULL) {
        goto FREE_MATRICES;
    }

    layer->b = malloc(y_size);
    if (layer->b == NULL) {
        goto FREE_MATRICES;
    }

    layer->dx = malloc(x_size);
    if (layer->dx == NULL) {
        goto FREE_MATRICES;
    }

    layer->dz = malloc(y_size);
    if (layer->dz == NULL) {
        goto FREE_MATRICES;
    }

    layer->dw = malloc(w_size);
    if (layer->dw == NULL) {
        goto FREE_MATRICES;
    }

    layer->db = malloc(y_size);
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

void scnn_layer_connect(scnn_layer* prev, scnn_layer* next)
{
    next->params.in = prev->params.out;
}

float *scnn_layer_forward(scnn_layer *layer, const float *x)
{
    if ((layer == NULL) || (x == NULL)) {
        return NULL;
    }

    scnn_scopy(layer->params.in, x, 1, layer->x, 1);

    const int m = 1; // Batch dimension
    const int n = layer->params.out;
    const int k = layer->params.in;

    // y = b: Broadcast for batch dimension
    for (int i = 0; i < m; i++) {
        scnn_scopy(
            layer->params.out, layer->b, 1, &layer->y[i * layer->params.out], 1
        );
    }

    // y = x * W + b
    scnn_sgemm(
        SCNN_BLAS_NO_TRANS, SCNN_BLAS_NO_TRANS,
        m, n, k,
        1.0, layer->x, k,
        layer->w, n,
        1.0, layer->y, n
    );

    // Activation
    sigmoid(layer->y, layer->z, layer->params.out);

    return layer->z;
}

float *scnn_layer_backward(scnn_layer *layer, const float *dy)
{
    if ((layer == NULL) || (dy == NULL)) {
        return NULL;
    }

    // dz = dy * z * (1 - z)
    for (int i = 0; i < layer->params.out; i++) {
        layer->dz[i] = dy[i] * layer->z[i] * (1.0f - layer->z[i]);
    }

    // dx = 0
    for (int i = 0; i < layer->params.in; i++) {
        layer->dx[i] = 0;
    }

    int m = 1; // Batch dimension
    int n = layer->params.in;
    int k = layer->params.out;

    // dx = dy * WT
    scnn_sgemm(
        SCNN_BLAS_NO_TRANS, SCNN_BLAS_TRANS,
        m, n, k,
        1.0, layer->dz, k,
        layer->w, k,
        1.0, layer->dx, n
    );

    // dw = 0
    for (int i = 0; i < layer->params.in * layer->params.out; i++) {
        layer->dw[i] = 0;
    }

    m = layer->params.in;
    n = layer->params.out;
    k = 1;

    // dW = xT * dy
    scnn_sgemm(
        SCNN_BLAS_TRANS, SCNN_BLAS_NO_TRANS,
        m, n, k,
        1.0, layer->x, m,
        layer->dz, n,
        1.0, layer->dw, n
    );

    // db = 0
    for (int i = 0; i < layer->params.out; i++) {
        layer->db[i] = 0;
    }

    // db = dy / (batch size):
    // Broadcast for batch dimension
    for (int i = 0; i < 1; i++) {
        scnn_saxpy(layer->params.out, 1, &layer->dz[i * layer->params.out], 1, layer->db, 1);
    }

    return layer->dx;
}

void layer_update(scnn_layer *layer, const float learning_rate)
{
    const int w_size = layer->params.in * layer->params.out;

    scnn_saxpy(w_size, -learning_rate, layer->dw, 1, layer->w, 1);

    scnn_saxpy(layer->params.out, -learning_rate, layer->db, 1, layer->b, 1);
}

void scnn_layer_free(scnn_layer **layer)
{
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
