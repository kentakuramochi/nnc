/**
 * @file nn_loss.c
 * @brief Loss function
 *
 */
#include "nn_loss.h"

float nn_loss(const int type, const float *y, const float *t, const size_t size) {
    return loss_forwards[type](y, t, size);
}

void nn_loss_backward(
    float *diff, const int type, const float *y, const float *t, const size_t size
) {
    loss_backwards[type](diff, y, t, size);
}
