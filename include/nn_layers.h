/**
 * @file nn_layers.h
 * @brief Interface to network layers
 *
 */
#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include <stddef.h>

#include "nn_layer.h"

/**
 * @brief Initialization functions for each layer
 */
NnLayer* (*nn_layer_init_funcs[])(NnLayer*) = {
    NULL,
    identity_init
};

#endif // NN_LAYERS_H
