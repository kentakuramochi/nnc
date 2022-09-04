/**
 * @file scnn_sigmoid.c
 * @brief Sigmoid layer
 * 
 */
#include <stddef.h>
#include <math.h>

#include "scnn_sigmoid.h"
#include "scnn_mat.h"
#include "scnn_blas.h"

/**
 * @brief Set the matrix size
 * 
 * @param[in] n Batch size N
 * @param[in] c Channel size C
 * @param[in] h Height H
 * @param[in] w Width W
 */
static void set_size(struct scnn_layer *self, const int n, const int c, const int h, const int w)
{
    if (self == NULL) {
        return;
    }

    if ((n < 1) || (c < 1) || (h < 1) || (w < 1)) {
        return;
    }

    // input channels = (C*H*W) in input matrix
    if ((c * h * w) != self->params.in) {
        return;
    }

    scnn_mat_init(&self->x, n, c, 1, 1);
    scnn_mat_init(&self->y, n, c, 1, 1);

    scnn_mat_init(&self->dx, self->x.n, self->x.c, self->x.h, self->x.w);
}

/**
 * @brief Forward propagation
 * 
 * @param[in,out] self  Pointer to target layer
 * @param[in]     x     Input matrix
 */
static void forward(scnn_layer *self, scnn_mat *x)
{
    if ((self == NULL) || (x == NULL)) {
        return;
    }

    scnn_scopy(self->x.size, x->data, 1, self->x.data, 1);
    for (int i = 0; i < self->x.size; i++) {
        self->y.data[i] = 1.0 / (1 + exp(-self->x.data[i]));
    }
}

/**
 * @brief Backward propagation
 * 
 * @param[in,out] self  Pointer to layer
 * @param[in]     dy    Diffirential of output matrix
 */
static void backward(scnn_layer *self, scnn_mat *dy)
{
    if ((self == NULL) || (dy == NULL)) {
        return;
    }

    for (int i = 0; i < self->y.size; i++) {
        self->dx.data[i] = dy->data[i] * (1.0f - self->y.data[i]) * self->y.data[i];
    }
}

scnn_layer *scnn_sigmoid_layer(const scnn_layer_params params)
{
    if (params.in < 1) {
        return NULL;
    }

    scnn_layer *layer = scnn_layer_alloc();
    if (layer == NULL) {
        return NULL;
    }

    // (input size) == (output size)
    layer->params.in  = params.in;
    layer->params.out = params.in;

    layer->forward  = forward;
    layer->backward = backward;

    layer->set_size = set_size;

    return layer;
}
