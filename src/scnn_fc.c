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

    scnn_mat_init(&self->dx, self->x.n, self->x.c, self->x.h, self->x.w);
    scnn_mat_init(&self->dw, self->w.n, self->w.c, self->w.h, self->w.w);
    scnn_mat_init(&self->db, self->b.n, self->b.c, self->b.h, self->b.w);
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
        self->x.n, self->w.n, self->x.c,
        1.0, self->x.data, self->x.c,
        self->w.data, self->w.n, 1.0,
        self->y.data, self->y.c);
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
        dy->n, self->w.c, dy->c,
        1.0, dy->data, dy->c,
        self->w.data, self->w.n, 1.0,
        self->dx.data, self->dx.c);

    // dW = x^T dy
    scnn_mat_fill(&self->dw, 0);
    scnn_sgemm(SCNN_BLAS_TRANS, SCNN_BLAS_NO_TRANS,
        self->x.c, dy->c, self->x.n,
        1.0, self->x.data, self->x.c,
        dy->data, dy->c, 1.0,
        self->dw.data, self->dw.n);

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
