/**
 * @file activation.c
 * @brief Activation function
 *
 */
#include "activation.h"

#include <float.h>
#include <math.h>

float *sigmoid(const float *x, float *y, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        y[i] = 1.0 / (1 + expf(-x[i]));
    }

    return y;
}

float *softmax(const float *x, float *y, const size_t size) {
    float max = -FLT_MAX;
    for (size_t i = 0; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    // Subtract a max of the input to avoid overflow
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += expf(x[i] - max);
    }

    for (size_t i = 0; i < size; i++) {
        y[i] = expf(x[i] - max) / sum;
    }

    return y;
}
