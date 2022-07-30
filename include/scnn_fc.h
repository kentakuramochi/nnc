/**
 * @file scnn_fc.h
 * @brief Fully connected layer
 * 
 */
#ifndef SCNN_FC_H
#define SCNN_FC_H

#include "scnn_layer.h"

/**
 * @brief Allocate fully connected layer
 * 
 * @param[in] params    Layer parameters
 * @return              Pointer to layer, NULL if failed
 */
scnn_layer *scnn_fc_layer(const scnn_layer_params params);

#endif // SCNN_FC_H
