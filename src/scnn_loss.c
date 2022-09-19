/**
 * @file scnn_loss.c
 * @brief Loss function
 * 
 */
#include "scnn_loss.h"

#include <math.h>

float scnn_mean_squared_error(const float *y, const float *t, const int size)
{
    float err = 0;

    for (int i = 0; i < size; i++) {
        err += (t[i] - y[i]) * (t[i] - y[i]);
    }

    return 0.5 * err;
}

float scnn_cross_entropy_loss(const float *y, const float *t, const int size)
{
    // small constant to avoid log(0)
    const float epsilon = 1e-7;

    float err = 0;

    for (int i = 0; i < size; i++) {
        err += t[i] * log(y[i] + epsilon);
    }

    return -err;
}
