/**
 * @file scnn_fc.c
 * @brief Fully connected layer
 * 
 */
#include <stddef.h>

#include "scnn_fc.h"
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
    scnn_mat_init(&self->y, n, self->params.out, 1, 1);
    scnn_mat_init(&self->w, self->params.out, self->params.in, 1, 1);
    scnn_mat_init(&self->b, n, self->params.out, 1, 1);

    scnn_mat_init(&self->dx, self->x.shape.d[0], self->x.shape.d[1], self->x.shape.d[2], self->x.shape.d[3]);
    scnn_mat_init(&self->dw, self->w.shape.d[0], self->w.shape.d[1], self->w.shape.d[2], self->w.shape.d[3]);
    scnn_mat_init(&self->db, self->b.shape.d[0], self->b.shape.d[1], self->b.shape.d[2], self->b.shape.d[3]);
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

    scnn_scopy(self->y.size, self->b.data, 1, self->y.data, 1);
    scnn_sgemm(SCNN_BLAS_NO_TRANS, SCNN_BLAS_NO_TRANS,
        self->x.shape.d[0], self->w.shape.d[0], self->x.shape.d[1],
        1.0, self->x.data, self->x.shape.d[1],
        self->w.data, self->w.shape.d[0], 1.0,
        self->y.data, self->y.shape.d[1]);
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

    // dx = dy W^T
    scnn_mat_fill(&self->dx, 0);
    scnn_sgemm(SCNN_BLAS_NO_TRANS, SCNN_BLAS_TRANS,
        dy->shape.d[0], self->w.shape.d[1], dy->shape.d[1],
        1.0, dy->data, dy->shape.d[1],
        self->w.data, self->w.shape.d[0], 1.0,
        self->dx.data, self->dx.shape.d[1]);

    // dW = x^T dy
    scnn_mat_fill(&self->dw, 0);
    scnn_sgemm(SCNN_BLAS_TRANS, SCNN_BLAS_NO_TRANS,
        self->x.shape.d[1], dy->shape.d[1], self->x.shape.d[0],
        1.0, self->x.data, self->x.shape.d[1],
        dy->data, dy->shape.d[1], 1.0,
        self->dw.data, self->dw.shape.d[0]);

    // db = dy
    scnn_scopy(self->db.size, dy->data, 1, self->db.data, 1);
}

scnn_layer *scnn_fc_layer(const scnn_layer_params params)
{
    if ((params.in < 1) || (params.out < 1)) {
        return NULL;
    }

    scnn_layer *layer = scnn_layer_alloc(params);
    if (layer == NULL) {
        return NULL;
    }

    layer->params.in  = params.in;
    layer->params.out = params.out;

    layer->forward  = forward;
    layer->backward = backward;

    layer->set_size = set_size;

    return layer;
}
