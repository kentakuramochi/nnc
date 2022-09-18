/**
 * @file scnn_softmax.h
 * @brief Softmax layer
 * 
 */
#ifndef SCNN_SOFTMAX_H
#define SCNN_SOFTMAX_H

#include "scnn_layer.h"

/**
 * @brief Allocate softmax layer
 * 
 * @param[in] params    Layer parameters
 * @return              Pointer to layer, NULL if failed
 */
scnn_layer *scnn_softmax_layer(const scnn_layer_params params);

#endif // SCNN_SOFTMAX_H
