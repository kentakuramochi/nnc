/**
 * @file scnn_softmax.c
 * @brief Softmax layer
 * 
 */
#include <stddef.h>
#include <math.h>

#include "scnn_softmax.h"
#include "scnn_blas.h"

/**
 * @brief Initialize Softmax layer
 * 
 * @param[in,out] self  Softmax layer
 * @return              Pointer to initialized layer, NULL if failed
 */
static scnn_layer *init(struct scnn_layer *self)
{
    if (self == NULL) {
        return NULL;
    }

    self->x = scnn_mat_alloc(self->params.in_shape);
    if (self->x == NULL) {
        return NULL;
    }
    self->y = scnn_mat_alloc(self->params.in_shape);
    if (self->y == NULL) {
        goto FREE_X;
    }

    self->dx = scnn_mat_alloc(self->x->shape);
    if (self->dx == NULL) {
        goto FREE_Y;
    }

    return self;

FREE_Y:
    scnn_mat_free(&self->y);
FREE_X:
    scnn_mat_free(&self->x);

    return NULL;
}

/**
 * @brief Forward propagation
 * 
 * @param[in,out] self  Sigmoid layer
 * @param[in]     x     Input matrix
 */
static void forward(scnn_layer *self, scnn_dtype *x)
{
    if ((self == NULL) || (x == NULL)) {
        return;
    }

    scnn_scopy(self->x->size, x, 1, self->x->data, 1);

    // NCHW order
    // shape[0] (N): batch dimension
    // shape[1] (C): axis
    const int dim_c  = self->x->shape[1] * self->x->shape[2] * self->x->shape[3];
    const int dim_xy = self->x->shape[2] * self->x->shape[3];

    scnn_dtype *px = self->x->data;
    scnn_dtype *py = self->y->data;

    for (int n = 0; n < self->x->shape[0]; n++) {
        for (int m = 0; m < dim_xy; m++) {
            scnn_dtype sum = 0;
            for (int i = 0; i < self->x->shape[1]; i++) {
                sum += exp(px[i * dim_xy]);
            }
            for (int i = 0; i < self->x->shape[1]; i++) {
                py[i * dim_xy] = exp(px[i * dim_xy]) / sum;
            }
            px++;
            py++;
        }
        px += (dim_c - dim_xy);
        py += (dim_c - dim_xy);
    }
}

/**
 * @brief Backward propagation
 * 
 * @param[in,out] self  Pointer to layer
 * @param[in]     dy    Diffirential of output matrix
 */
static void backward(scnn_layer *self, scnn_dtype *dy)
{
    if ((self == NULL) || (dy == NULL)) {
        return;
    }

    // backward propagation with cross entropy loss
    for (int i = 0; i < self->y->size; i++) {
        self->dx->data[i] = dy[i];
    }
}

scnn_layer *scnn_softmax_layer(const scnn_layer_params params)
{
    scnn_layer *layer = scnn_layer_alloc(params);
    if (layer == NULL) {
        return NULL;
    }

    layer->init = init;

    layer->forward  = forward;
    layer->backward = backward;

    return layer;
}
