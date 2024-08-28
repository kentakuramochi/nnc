/**
 * @file bce_loss.c
 * @brief Binary cross entropy loss
 *
 */
#include "bce_loss.h"

#include <math.h>

float bce_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        loss += t[i] * log2f(y[i]) + (1.0f - t[i]) * log2f(1.0f - y[i]);
    }

    loss /= size;

    return -loss;
}

void bce_loss_backward(float *diff, const float *y, const float *t, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        diff[i] = -(t[i] / y[i] - (1.0f - t[i]) / (1.0f - y[i])) / (size * logf(2));  // log2(x)=ln(x)/ln(2)
    }
}
