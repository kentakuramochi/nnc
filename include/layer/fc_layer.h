/**
 * @file fc_layer.h
 * @brief Fully connected layer
 *
 */
#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"

/**
 * @brief Allocate a fully connected layer
 * 
 * @param[in,out] layer Pointer to a layer
 * @return Pointer to the layer, NULL if failed
*/
Layer *fc_layer_init(Layer *layer);

#endif // FC_LAYER_H
