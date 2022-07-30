/**
 * @file scnn_fc.c
 * @brief Fully connected layer
 * 
 */
#include <stddef.h>
// stub
#include <assert.h>

#include "scnn_fc.h"

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
    // stub
    assert(self != NULL);
    assert(n != 0);
    assert(c != 0);
    assert(h != 0);
    assert(w != 0);
}

/**
 * @brief Forward propagation
 * 
 * @param[in,out] self  Pointer to target layer
 * @param[in]     x     Input matrix
 */
static void forward(scnn_layer *self, scnn_mat *x)
{
    // stub
    assert(self != NULL);
    assert(x != NULL);
}

/**
 * @brief Backward propagation
 * 
 * @param[in,out] self  Pointer to layer
 * @param[in]     dy    Diffirential of output matrix
 */
static void backward(scnn_layer *self, scnn_mat *dy)
{
    // stub
    assert(self != NULL);
    assert(dy != NULL);
}

scnn_layer *scnn_fc_layer(const scnn_layer_params params)
{
    if ((params.in < 1) || (params.out < 1)) {
        return NULL;
    }

    scnn_layer *layer = scnn_layer_alloc();
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
