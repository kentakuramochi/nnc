/**
 * @file scnn_sigmoid.c
 * @brief Sigmoid layer
 * 
 */
#include <stddef.h>
#include <math.h>

#include "scnn_sigmoid.h"
#include "scnn_blas.h"

/**
 * @brief Initialize Sigmoid layer
 * 
 * @param[in,out] self  Sigmoid layer
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
    for (int i = 0; i < self->x->size; i++) {
        self->y->data[i] = 1.0 / (1 + exp(-self->x->data[i]));
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

    for (int i = 0; i < self->y->size; i++) {
        self->dx->data[i] = dy[i] * (1.0f - self->y->data[i]) * self->y->data[i];
    }
}

scnn_layer *scnn_sigmoid_layer(const scnn_layer_params params)
{
    scnn_layer *layer = scnn_layer_alloc(params);
    if (layer == NULL) {
        return NULL;
    }

    layer->params.type = SCNN_LAYER_SIGMOID;

    layer->init = init;

    layer->forward  = forward;
    layer->backward = backward;

    return layer;
}
