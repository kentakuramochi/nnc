/**
 * @file layers.h
 * @brief Interface to network layers
 *
 */
#ifndef LAYERS_H
#define LAYERS_H

#include <stddef.h>

#include "layer.h"

#include "layer/fc_layer.h"
#include "layer/sigmoid_layer.h"

/**
 * @brief Initialization functions for each layer
 */
Layer* (*layer_init_funcs[])(Layer*) = {
    NULL,
    fc_layer_init,
    sigmoid_layer_init
};

#endif // LAYERS_H
