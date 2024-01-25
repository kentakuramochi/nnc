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

typedef struct scnn_layer scnn_layer;

/**
 * @brief Get an output of a layer
 * 
 * @param[in]   layer   Layer
 * @return              Pointer to an output of the layer
*/
scnn_dtype *scnn_layer_y(const scnn_layer *layer);

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
