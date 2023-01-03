/**
 * @file scnn_layer.h
 * @brief Layer structure
 * 
 */
#ifndef SCNN_LAYER_H
#define SCNN_LAYER_H

#include "scnn_mat.h"

/**
 * @brief Layer type
 * 
 */
typedef enum scnn_layer_type {
    SCNN_LAYER_NONE,    //!< None type
    SCNN_LAYER_FC,      //!< FullyConnected layer
    SCNN_LAYER_SIGMOID, //!< Sigmoid layer
    SCNN_LAYER_SOFTMAX  //!< Softmax layer
} scnn_layer_type;

/**
 * @brief Layer parameters
 * 
 */
typedef struct scnn_layer_params {
    scnn_layer_type type;       //!< Layer type
    int in_shape[SCNN_MAT_DIM]; //!< Input shape
    int out;                    //!< Output channels
    int id;                     //!< ID of a layer
    int prev_id;                //!< ID of the previous layer
    int next_id;                //!< ID of the next layer
} scnn_layer_params;

/**
 * @brief Layer structure
 * 
 */
typedef struct scnn_layer {
    scnn_layer_params params;   //!< Layer parameters

    scnn_mat* x;     //!< Input matrix
    scnn_mat* y;     //!< Output matrix
    scnn_mat* w;     //!< Weight matrix
    scnn_mat* b;     //!< Bias matrix

    scnn_mat* dx;    //!< Difference of input matrix
    scnn_mat* dw;    //!< Difference of weight matrix
    scnn_mat* db;    //!< Difference of bias matrix

    struct scnn_layer* (*init)(struct scnn_layer *self);  //!< Initialize layer

    void (*forward)(struct scnn_layer *self, scnn_dtype* x);     //!< Forward propagation
    void (*backward)(struct scnn_layer *self, scnn_dtype* dy);   //!< Backward propagation
} scnn_layer;

/**
 * @brief Allocate layer
 * 
 * @return Pointer to layer, NULL if failed
 */
scnn_layer *scnn_layer_alloc(const scnn_layer_params params);

/**
 * @brief Free layer
 * 
 * @param[in,out] layer Pointer to pointer of layer
 */
void scnn_layer_free(scnn_layer **layer);

#endif // SCNN_LAYER_H
