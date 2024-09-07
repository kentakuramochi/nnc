/**
 * @file bce_loss.c
 * @brief Binary cross entropy loss
 *
 */
#include "loss/bce_loss.h"

#include <math.h>

/**
 * @brief Binary cross entropy loss
 *
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
 * @return float Loss
*/
static float forward(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < size; i++) {
        loss += t[i] * logf(y[i]) + (1.0f - t[i]) * logf(1.0f - y[i]);
    }

    loss /= size;

    return -loss;
}

/**
 * @brief Backward of the BCE loss
 *
 * @param[out] diff Difference of loss function by the output
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
*/
static void backward(float *diff, const float *y, const float *t, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        diff[i] = -(t[i] / y[i] - (1.0f - t[i]) / (1.0f - y[i])) / size;
    }
}

LossFunc bce_loss(void) {
    LossFunc loss_func = {
        .forward = forward,
        .backward = backward 
    };
    return loss_func;
}
