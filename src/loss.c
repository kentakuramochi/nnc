/**
 * @file loss.c
 * @brief Loss function
 *
 */
#include "loss.h"

float mse_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (int i = 0; i < size; i++) {
        loss += (t[i] - y[i]) * (t[i] - y[i]);
    }

    loss /= size;

    return loss;
}
