/**
 * @file scnn_layer.h
 * @brief Layer structure
 * 
 */
#ifndef SCNN_LAYER_H
#define SCNN_LAYER_H

#include "scnn_mat.h"

/**
 * @brief Layer structure
 * 
 */
typedef struct scnn_layer {
    int id;         //<! Layer ID

    scnn_mat x;     //<! Input matrix
    scnn_mat y;     //<! Output matrix
    scnn_mat w;     //<! Weight matrix
    scnn_mat b;     //<! Bias matrix

    scnn_mat dx;    //<! Difference of input matrix
    scnn_mat dw;    //<! Difference of weight matrix
    scnn_mat db;    //<! Difference of bias matrix

    int prev_id;    //<! Layer ID of the previous
    int next_id;    //<! Layer ID of the next

    void (*forward)(struct scnn_layer *self, scnn_mat* x);     //<! Forward propagation
    void (*backward)(struct scnn_layer *self, scnn_mat* dy);   //<! Backward propagation
} scnn_layer;

/**
 * @brief Allocate layer
 * 
 * @return Pointer to layer, NULL if failed
 */
scnn_layer *scnn_layer_alloc(void);

/**
 * @brief Free layer
 * 
 * @param layer Pointer to pointer of layer
 */
void scnn_layer_free(scnn_layer **layer);

#endif // SCNN_LAYER_H
