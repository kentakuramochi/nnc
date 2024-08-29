/**
 * @file nn_losses.h
 * @brief Interface to loss functions
 *
 */
#ifndef NN_LOSSES_H
#define NN_LOSSES_H

#include <stddef.h>

#include "nn_loss.h"

#include "loss/bce_loss.h"

/**
 * @brief Table of loss functions
*/
float (*loss_forwards[])(const float*, const float*, const size_t) = {
    NULL,
    bce_loss
};

/**
 * @brief Table of loss function backwards
*/
void (*loss_backwards[])(float*, const float*, const float*, const size_t) = {
    NULL,
    bce_loss_backward
};

#endif // NN_LOSSES_H
