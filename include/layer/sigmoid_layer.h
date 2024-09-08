/**
 * @file sigmoid_layer.h
 * @brief Sigmoid layer
 *
 */
#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "layer.h"

/**
 * @brief Allocate a sigmoid layer
 * 
 * @param[in,out] layer Pointer to a layer
 * @return Pointer to the layer, NULL if failed
*/
Layer *sigmoid_layer_init(Layer *layer);

#endif // SIGMOID_LAYER_H
