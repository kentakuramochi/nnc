/**
 * @file loss.c
 * @brief Loss function
 *
 */
#include "loss.h"

#include <math.h>

float mse_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        loss += (t[i] - y[i]) * (t[i] - y[i]);
    }

    loss /= size;

    return loss;
}

float binary_cross_entropy_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        loss += t[i] * log2f(y[i]) + (1.0f - t[i]) * log2f(1.0f - y[i]);
    }

    loss /= size;

    return -loss;
}
