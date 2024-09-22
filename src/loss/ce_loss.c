/**
 * @file ce_loss.c
 * @brief Cross entropy loss
 */
#include "loss/ce_loss.h"

#include <math.h>

/**
 * @brief Binary cross entropy loss
 *
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] batch_size Batch size of data
 * @param[in] size Size of data
 * @return float Loss
 */
static float forward(const float *y, const float *t, const size_t batch_size, const size_t size) {
    float loss = 0.0f;

    for (size_t i = 0; i < batch_size; i++) {
        const float *b_y = &y[i * size];
        const float *b_t = &t[i * size];
        for (size_t j = 0; j < size; j++) {
            loss += b_t[j] * logf(b_y[j]);
        }
    }

    return -loss / (float)batch_size;
}

/**
 * @brief Backward of the BCE loss
 *
 * @param[out] grad Gradient of loss function by the output
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] batch_size Batch size of data
 * @param[in] size Size of data
 */
static void backward(float *grad, const float *y, const float *t, const size_t batch_size, const size_t size) {
    for (size_t i = 0; i < batch_size; i++) {
        const float *b_y = &y[i * size];
        const float *b_t = &t[i * size];
        float *b_grad = &grad[i * size];
        for (size_t j = 0; j < size; j++) {
            b_grad[j] = -b_t[j] / b_y[j] / batch_size;
        }
    }
}

LossFunc ce_loss(void) {
    LossFunc loss_func = {
        .forward = forward,
        .backward = backward 
    };
    return loss_func;
}
