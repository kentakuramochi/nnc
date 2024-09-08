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
            loss += b_t[j] * logf(b_y[j]) + (1.0f - b_t[j]) * logf(1.0f - b_y[j]);
        }
    }

    return -loss / (float)batch_size;
}

/**
 * @brief Backward of the BCE loss
 *
 * @param[out] diff Difference of loss function by the output
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] batch_size Batch size of data
 * @param[in] size Size of data
*/
static void backward(float *diff, const float *y, const float *t, const size_t batch_size, const size_t size) {
    for (size_t i = 0; i < batch_size; i++) {
        const float *b_y = &y[i * size];
        const float *b_t = &t[i * size];
        float *p_diff = &diff[i * size];
        for (size_t j = 0; j < size; j++) {
            p_diff[j] = -(b_t[j] / b_y[j] - (1.0f - b_t[j]) / (1.0f - b_y[j])) / (float)batch_size;
        }
    }
}

LossFunc bce_loss(void) {
    LossFunc loss_func = {
        .forward = forward,
        .backward = backward 
    };
    return loss_func;
}
