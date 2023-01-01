/**
 * @file scnn_fc.c
 * @brief Fully connected layer
 * 
 */
#include <stddef.h>

#include "scnn_fc.h"
#include "scnn_blas.h"

/**
 * @brief Initialize FC layer
 * 
 * @param[in,out] self  FC layer
 * @return              Pointer to intitialized layer, NULL if failed
 */
static struct scnn_layer* init(scnn_layer *self)
{
    if (self == NULL) {
        return NULL;
    }

    self->x = scnn_mat_alloc(self->params.in_shape);
    if (self->x == NULL) {
        return NULL;
    }
    self->y = scnn_mat_alloc(scnn_shape(self->params.in_shape[0], self->params.out, 1, 1));
    if (self->y == NULL) {
        goto FREE_X;
    }
    const int in = self->x->shape[1] * self->x->shape[2] * self->x->shape[3];
    self->w = scnn_mat_alloc(scnn_shape(in, self->params.out, 1, 1));
    if (self->w == NULL) {
        goto FREE_Y;
    }
    self->b = scnn_mat_alloc(scnn_shape(1, self->params.out, 1, 1));
    if (self->b == NULL) {
        goto FREE_W;
    }

    self->dx = scnn_mat_alloc(self->x->shape);
    if (self->dx == NULL) {
        goto FREE_B;
    }
    self->dw = scnn_mat_alloc(self->w->shape);
    if (self->dw == NULL) {
        goto FREE_DX;
    }
    self->db = scnn_mat_alloc(self->b->shape);
    if (self->db == NULL) {
        goto FREE_DW;
    }

    return self;

FREE_DW:
    scnn_mat_free(&self->dw);
FREE_DX:
    scnn_mat_free(&self->dx);
FREE_B:
    scnn_mat_free(&self->b);
FREE_W:
    scnn_mat_free(&self->w);
FREE_Y:
    scnn_mat_free(&self->y);
FREE_X:
    scnn_mat_free(&self->x);

    return NULL;
}

/**
 * @brief Forward propagation
 * 
 * @param[in,out] self  FC layer
 * @param[in]     x     Input matrix
 */
static void forward(scnn_layer *self, scnn_dtype *x)
{
    if ((self == NULL) || (x == NULL)) {
        return;
    }

    scnn_scopy(self->x->size, x, 1, self->x->data, 1);

    // y = xW+b
    const int m = self->x->shape[0];
    const int n = self->y->shape[1];
    const int k = self->x->shape[1];
    // broadcast for batch dimension
    for (int i = 0; i < m; i++) {
        scnn_scopy(self->b->size, self->b->data, 1, &self->y->data[i * self->b->size], 1);
    }
    scnn_sgemm(SCNN_BLAS_NO_TRANS, SCNN_BLAS_NO_TRANS,
        m, n, k,
        1.0, self->x->data, k,
        self->w->data, n,
        1.0, self->y->data, n);
}

/**
 * @brief Backward propagation
 * 
 * @param[in,out] self  FC layer
 * @param[in]     dy    Diffirential of output matrix
 */
static void backward(scnn_layer *self, scnn_dtype *dy)
{
    if ((self == NULL) || (dy == NULL)) {
        return;
    }

    // dx = dy W^T
    scnn_mat_fill(self->dx, 0);
    int m = self->x->shape[0];
    int n = self->x->shape[1];
    int k = self->y->shape[1];
    scnn_sgemm(SCNN_BLAS_NO_TRANS, SCNN_BLAS_TRANS,
        m, n, k,
        1.0, dy, k,
        self->w->data, k,
        1.0, self->dx->data, n);

    // dW = x^T dy
    scnn_mat_fill(self->dw, 0);
    m = self->x->shape[1];
    n = self->y->shape[1];
    k = self->y->shape[0];
    scnn_sgemm(SCNN_BLAS_TRANS, SCNN_BLAS_NO_TRANS,
        m, n, k,
        1.0, self->x->data, m,
        dy, n,
        1.0, self->dw->data, n);

    // db = dy / (batch size)
    scnn_mat_fill(self->db, 0);
    // broadcast for batch dimension
    const scnn_dtype batch_size = 1.0 / self->dx->shape[0];
    for (int i = 0; i < self->y->shape[0]; i++) {
        scnn_saxpy(self->db->size, batch_size, &dy[i * self->db->size], 1, self->db->data, 1);
    }
}

scnn_layer *scnn_fc_layer(const scnn_layer_params params)
{
    scnn_layer *layer = scnn_layer_alloc(params);
    if (layer == NULL) {
        return NULL;
    }

    layer->params.type = SCNN_LAYER_FC;

    layer->init = init;

    layer->forward  = forward;
    layer->backward = backward;

    return layer;
}
