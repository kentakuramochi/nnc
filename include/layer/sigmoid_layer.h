/**
 * @file sigmoid_layer.h
 * @brief Sigmoid layer
 *
 */
#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "nn_layer.h"

/**
 * @brief Allocate a sigmoid layer
 * 
 * @param[in,out] layer Pointer to a layer
 * @return Pointer to the layer, NULL if failed
*/
NnLayer *sigmoid_layer_init(NnLayer *layer);

#endif // SIGMOID_LAYER_H
