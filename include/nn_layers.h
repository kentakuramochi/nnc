/**
 * @file nn_layers.h
 * @brief Interface to network layers
 *
 */
#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include <stddef.h>

#include "nn_layer.h"

#include "layer/fc_layer.h"
#include "layer/sigmoid_layer.h"

/**
 * @brief Initialization functions for each layer
 */
NnLayer* (*nn_layer_init_funcs[NUM_LAYER_TYPE])(NnLayer*) = {
    NULL,
    fc_layer_init,
    sigmoid_layer_init
};

#endif // NN_LAYERS_H
