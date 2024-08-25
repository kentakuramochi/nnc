/**
 * @file fc_layer.h
 * @brief Fully connected layer
 *
 */
#ifndef FC_LAYER_H
#define FC_LAYER_H

#include <stddef.h>

#include "nn_layer.h"

// Temporal parameter for the layer type
#define NN_LAYER_TYPE_FC 1

/**
 * @brief Allocate a fully connected layer
 * 
 * @param[in,out] layer Pointer to a layer
 * @return Pointer to the layer, NULL if failed
*/
NnLayer *fc_layer_init(NnLayer *layer);

#endif // FC_LAYER_H
