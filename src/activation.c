/**
 * @file activation.c
 * @brief Activation function
 *
 */

#include "activation.h"

#include <math.h>

float *sigmoid(const float *x, float *y, const size_t size)
{
    for (int i = 0; i < size; i++) {
        y[i] = 1.0 / (1 + expf(-x[i]));
    }

    return y;
}
