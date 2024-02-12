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
    int in; //!< Number of input elements
    int out; //!< Number of output elements
} NnLayerParams;

typedef struct NnLayer {
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
} NnLayer;

/**
 * @brief Allocate a layer
 *
 * @param[in] params Parameters for a layer
 * @return Pointer to layer, NULL if failed
 */
NnLayer *nn_layer_alloc(const NnLayerParams params);

/**
 * @brief Initialze a layer
 *
 * @param[in,out] layer Layer
 * @return Pointer the the layer, NULL if failed
 */
NnLayer *nn_layer_init(NnLayer *layer);

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

/**
 * @brief Free layer
 *
 * @param[in,out] layer Pointer to pointer of layer
 */
void nn_layer_free(NnLayer **layer);

#endif // NN_LAYER_H
