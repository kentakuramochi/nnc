/**
 * @file scnn_layer.h
 * @brief Layer structure
 * 
 */
#ifndef SCNN_LAYER_H
#define SCNN_LAYER_H

#include "scnn_mat.h"

/**
 * @brief Layer parameters
 * 
 */
typedef struct scnn_layer_params {
    int in;                 //!< Input elements
    int out;                //!< Output elements
} scnn_layer_params;

typedef struct scnn_layer {
    scnn_layer_params params;   //!< Layer parameters

    scnn_dtype* x;  //!< Input matrix
    scnn_dtype* y;  //!< Output matrix
    scnn_dtype* w;  //!< Weight matrix
    scnn_dtype* b;  //!< Bias matrix

    scnn_dtype* dx; //!< Difference of input matrix
    scnn_dtype* dw; //!< Difference of weight matrix
    scnn_dtype* db; //!< Difference of bias matrix
} scnn_layer;

/**
 * @brief Allocate a layer
 * 
 * @param[in]   params  Parameters for a layer
 * @return              Pointer to layer, NULL if failed
 */
scnn_layer *scnn_layer_alloc(const scnn_layer_params params);

/**
 * @brief Initialze a layer
 * 
 * @param[in,out]   layer   Layer
 * @return                  Pointer the the layer, NULL if failed
 */
scnn_layer *scnn_layer_init(scnn_layer* layer);

/**
 * @brief Connect 2 layers
 * 
 * @param[in,out]   prev    Previous layer, being connected from the next
 * @param[in,out]   next    Next layer, connect to the previous
 */
void scnn_layer_connect(scnn_layer* prev, scnn_layer* next);

/**
 * @brief Forward propagation of a layer
 * 
 * @param[in,out]   layer   Layer
 * @param[in]       x       An input of the layer
 * @return                  Pointer to the layer output
*/
scnn_dtype *scnn_layer_forward(scnn_layer *layer, const scnn_dtype *x);

/**
 * @brief Backward propagation of a layer
 * 
 * @param[in,out]   layer   Layer
 * @param[in]       dy      A differential of previous layer
 * @return                  Pointer to differential of an input of the layer
*/
scnn_dtype *scnn_layer_backward(scnn_layer *layer, const scnn_dtype *dy);

/**
 * @brief Free layer
 * 
 * @param[in,out] layer Pointer to pointer of layer
 */
void scnn_layer_free(scnn_layer **layer);

#endif // SCNN_LAYER_H
