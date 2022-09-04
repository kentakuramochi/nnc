/**
 * @file scnn_sigmoid.h
 * @brief Sigmoid layer
 * 
 */
#ifndef SCNN_SIGMOID_H
#define SCNN_SIGMOID_H

#include "scnn_layer.h"

/**
 * @brief Allocate Sigmoid connected layer
 * 
 * @param[in] params    Layer parameters
 * @return              Pointer to layer, NULL if failed
 */
scnn_layer *scnn_sigmoid_layer(const scnn_layer_params params);

#endif // SCNN_SIGMOID_H
