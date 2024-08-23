/**
 * @file nn_loss.c
 * @brief Loss function
 *
 */
#include "nn_loss.h"

#include <math.h>

/**
 * @brief Table of loss functions
*/
float (*loss_forward[])(const float*, const float*, const size_t) = {
    NULL,
    mse_loss
};

/**
 * @brief Table of loss function backwardings
*/
void (*loss_backward[])(float*, const float*, const float*, const size_t) = {
    NULL,
    mse_loss_backward
};

float nn_loss(const int type, const float *y, const float *t, const size_t size) {
    return loss_forward[type](y, t, size);
}

void nn_loss_backward(
    float *diff, const int type, const float *y, const float *t, const size_t size
) {
    loss_backward[type](diff, y, t, size);
}

float mse_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        loss += (t[i] - y[i]) * (t[i] - y[i]);
    }

    loss /= size;

    return loss;
}

void mse_loss_backward(
    float *diff, const float *y, const float *t, const size_t size
) {
    // dE/dy based on the computational graph
    for (size_t i = 0; i < size; i++) {
        diff[i] = 2 * (y[i] - t[i]) / size;
    }
}

float binary_cross_entropy_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        loss += t[i] * log2f(y[i]) + (1.0f - t[i]) * log2f(1.0f - y[i]);
    }

    loss /= size;

    return -loss;
}
