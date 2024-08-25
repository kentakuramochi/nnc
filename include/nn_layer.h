/**
 * @file nn_layer.h
 * @brief Layer structure
 *
 */
#ifndef NN_LAYER_H
#define NN_LAYER_H

#include <stddef.h>

// Dummy parameters for layer type
#define NN_LAYER_TYPE_NONE 0
#define NN_LAYER_TYPE_IDENTITY 1
// Num of layer types
#define NUM_LAYER_TYPE 3

/**
 * @brief Layer parameters
 *
 */
typedef struct NnLayerParams {
    int type; //!< Layer type
    int batch_size; //!< Number of batches
    int in; //!< Number of input elements
    int out; //!< Number of output elements
} NnLayerParams;

typedef struct NnLayer {
    NnLayerParams params;  //!< Layer parameters

    float *x; //!< Input matrix
    float *y; //!< Output matrix
    float *w; //!< Weight matrix
    float *b; //!< Bias matrix

    float *dx; //!< Difference of input matrix
    float *dw; //!< Difference of weight matrix
    float *db; //!< Difference of bias matrix

    float* (*forward)(struct NnLayer*, const float*);   //!< Forward
    float* (*backward)(struct NnLayer*, const float*);  //!< Backward
} NnLayer;

/**
 * @brief Allocate layer parameters
 *
 * @param[in,out] layer Pointer to a layer
 * @return Pointer to the layer, NULL if failed
 */
NnLayer *nn_layer_alloc_params(NnLayer *layer);

/**
 * @brief Free layer parameters
 *
 * @param[in,out] layer Pointer to a layer
 */
void nn_layer_free_params(NnLayer *layer);

/**
 * @brief Connect 2 layers
 *
 * @param[in,out] prev Previous layer, being connected from the next one
 * @param[in,out] next Next layer, connect to the previous one
 */
void nn_layer_connect(NnLayer *prev, NnLayer *next);

/**
 * @brief Forward propagation of a layer
 *
 * @param[in,out] layer Layer
 * @param[in] x An input of the layer
 * @return Pointer to the layer output
 */
float *nn_layer_forward(NnLayer *layer, const float *x);

/**
 * @brief Backward propagation of a layer
 *
 * @param[in,out] layer Layer
 * @param[in] dy A differential of previous layer
 * @return Pointer to differential of an input of the layer
 */
float *nn_layer_backward(NnLayer *layer, const float *dy);

/**
 * @brief Clear current gradients of layer
 *
 * @param[in,out] layer Pointer to the layer
 */
void nn_layer_clear_grad(NnLayer *layer);

/**
 * @brief Initialization functions for each layer
 */
extern NnLayer* (*nn_layer_init_funcs[NUM_LAYER_TYPE])(NnLayer*);

#endif // NN_LAYER_H
