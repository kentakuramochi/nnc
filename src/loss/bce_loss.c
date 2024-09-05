/**
 * @file bce_loss.c
 * @brief Binary cross entropy loss
 *
 */
#include "loss/bce_loss.h"

#include <math.h>

float bce_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        loss += t[i] * logf(y[i]) + (1.0f - t[i]) * logf(1.0f - y[i]);
    }

    loss /= size;

    return -loss;
}

void bce_loss_backward(float *diff, const float *y, const float *t, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        diff[i] = -(t[i] / y[i] - (1.0f - t[i]) / (1.0f - y[i])) / size;
    }
}
