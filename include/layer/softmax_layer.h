/**
 * @file softmax_layer.h
 * @brief Softmax layer
 */
#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"

/**
 * @brief Allocate a softmax layer
 * 
 * @param[in,out] layer Pointer to a layer
 * @return Pointer to the layer, NULL if failed
 */
Layer *softmax_layer_init(Layer *layer);

#endif // SOFTMAX_LAYER_H
