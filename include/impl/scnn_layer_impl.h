/**
 * @file scnn_layer_impl.h
 * @brief Implementaion of layer structure
 * 
 */
#ifndef SCNN_LAYER_IMPL_H
#define SCNN_LAYER_IMPL_H

#include "scnn_layer.h"

/**
 * @brief Layer structure
 * 
 */
struct scnn_layer {
    scnn_layer_params params;   //!< Layer parameters

    scnn_mat* x;     //!< Input matrix
    scnn_mat* y;     //!< Output matrix
    scnn_mat* w;     //!< Weight matrix
    scnn_mat* b;     //!< Bias matrix

    scnn_mat* dx;    //!< Difference of input matrix
    scnn_mat* dw;    //!< Difference of weight matrix
    scnn_mat* db;    //!< Difference of bias matrix

    struct scnn_layer* (*init)(struct scnn_layer *self);  //!< Initialize layer

    scnn_dtype* (*forward)(struct scnn_layer *self, const scnn_dtype* x);     //!< Forward propagation
    scnn_dtype* (*backward)(struct scnn_layer *self, const scnn_dtype* dy);   //!< Backward propagation
};

#endif // SCNN_LAYER_IMPL_H
