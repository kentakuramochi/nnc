/**
 * @file scnn_layer.h
 * @brief Layer structure
 *
 */
#ifndef SCNN_LAYER_H
#define SCNN_LAYER_H

/**
 * @brief Layer parameters
 *
 */
typedef struct scnn_layer_params {
    int in; //!< Number of input elements
    int out; //!< Number of output elements
} scnn_layer_params;

typedef struct scnn_layer {
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
} scnn_layer;

/**
 * @brief Allocate a layer
 *
 * @param[in] params Parameters for a layer
 * @return Pointer to layer, NULL if failed
 */
scnn_layer *scnn_layer_alloc(const scnn_layer_params params);

/**
 * @brief Initialze a layer
 *
 * @param[in,out] layer Layer
 * @return Pointer the the layer, NULL if failed
 */
scnn_layer *scnn_layer_init(scnn_layer *layer);

/**
 * @brief Connect 2 layers
 *
 * @param[in,out] prev Previous layer, being connected from the next
 * @param[in,out] next Next layer, connect to the previous
 */
void scnn_layer_connect(scnn_layer *prev, scnn_layer *next);

/**
 * @brief Forward propagation of a layer
 *
 * @param[in,out] layer Layer
 * @param[in] x An input of the layer
 * @return Pointer to the layer output
 */
float *scnn_layer_forward(scnn_layer *layer, const float *x);

/**
 * @brief Backward propagation of a layer
 *
 * @param[in,out] layer Layer
 * @param[in] dy A differential of previous layer
 * @return Pointer to differential of an input of the layer
 */
float *scnn_layer_backward(scnn_layer *layer, const float *dy);


/**
 * @brief Update layer parameter
 *
 * @param[in,out] layer Pointer to the layer
 * @param[in] learning_rate Learning rate
 */
void layer_update(scnn_layer *layer, const float learning_rate);

/**
 * @brief Free layer
 *
 * @param[in,out] layer Pointer to pointer of layer
 */
void scnn_layer_free(scnn_layer **layer);

#endif // SCNN_LAYER_H
