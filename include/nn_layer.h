/**
 * @file nn_layer.h
 * @brief Layer structure
 *
 */
#ifndef NN_LAYER_H
#define NN_LAYER_H

/**
 * @brief Layer parameters
 *
 */
typedef struct NnLayerParams {
    int batch_size; //!< Number of batches
    int in; //!< Number of input elements
    int out; //!< Number of output elements
} NnLayerParams;

typedef struct NnLayer {
    int batch_size; //!< Number of batches
    int in; //!< Number of input elements
    int out; //!< Number of output elements
    float *x; //!< Input matrix
    float *y; //!< Output matrix
    float *z; //!< Activation output matrix
    float *w; //!< Weight matrix
    float *b; //!< Bias matrix
    float *dx; //!< Difference of input matrix
    float *dz; //!< Difference of activation
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
 * @param[in,out] prev Previous layer, being connected from the next
 * @param[in,out] next Next layer, connect to the previous
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
 * @brief Update layer parameter
 *
 * @param[in,out] layer Pointer to the layer
 * @param[in] learning_rate Learning rate
 */
void nn_layer_update(NnLayer *layer, const float learning_rate);

// Temporal implementation of forward/backward prop
// TODO: migrate to an implementation of the specific layer
float* forward(NnLayer *layer, const float *x);
float *backward(NnLayer *layer, const float *dy);

#endif // NN_LAYER_H
