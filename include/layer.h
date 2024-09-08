/**
 * @file layer.h
 * @brief Layer structure
 *
 */
#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Type of layers
*/
typedef enum LayerType {
    LAYER_TYPE_NONE, //!< None
    LAYER_TYPE_FC, //!< Fully connected layer
    LAYER_TYPE_SIGMOID //!< Sigmoid layer
} LayerType;

/**
 * @brief Layer parameters
 *
 */
typedef struct LayerParams {
    LayerType type; //!< Layer type
    int batch_size; //!< Number of batches
    int in; //!< Number of input elements
    int out; //!< Number of output elements
} LayerParams;

/**
 * @brief Layer structure
*/
typedef struct Layer {
    LayerParams params;  //!< Layer parameters

    float *x; //!< Input matrix
    float *y; //!< Output matrix
    float *w; //!< Weight matrix
    float *b; //!< Bias matrix

    float *dx; //!< Difference of input matrix
    float *dw; //!< Difference of weight matrix
    float *db; //!< Difference of bias matrix

    float* (*forward)(struct Layer*, const float*);   //!< Forward
    float* (*backward)(struct Layer*, const float*);  //!< Backward
} Layer;

/**
 * @brief Allocate layer parameters
 *
 * @param[in,out] layer Pointer to a layer
 * @return Pointer to the layer, NULL if failed
 */
Layer *layer_alloc_params(Layer *layer);

/**
 * @brief Free layer parameters
 *
 * @param[in,out] layer Pointer to a layer
 */
void layer_free_params(Layer *layer);

/**
 * @brief Connect 2 layers
 *
 * @param[in,out] prev Previous layer, being connected from the next one
 * @param[in,out] next Next layer, connect to the previous one
 * @return true if 2 layers are connected, otherwise false
 */
bool layer_connect(Layer *prev, Layer *next);

/**
 * @brief Forward propagation of a layer
 *
 * @param[in,out] layer Layer
 * @param[in] x An input of the layer
 * @return Pointer to the layer output
 */
float *layer_forward(Layer *layer, const float *x);

/**
 * @brief Backward propagation of a layer
 *
 * @param[in,out] layer Layer
 * @param[in] dy A differential of previous layer
 * @return Pointer to differential of an input of the layer
 */
float *layer_backward(Layer *layer, const float *dy);

/**
 * @brief Clear current gradients of layer
 *
 * @param[in,out] layer Pointer to the layer
 */
void layer_clear_grad(Layer *layer);

/**
 * @brief Initialization functions for each layer
 */
extern Layer* (*layer_init_funcs[])(Layer*);

#endif // LAYER_H
