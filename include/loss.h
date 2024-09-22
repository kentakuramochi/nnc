/**
 * @file loss.h
 * @brief Loss function
 */
#ifndef LOSS_H
#define LOSS_H

#include <stddef.h>

/**
 * @brief Loss funcion interface
 */
typedef struct LossFunc {
    /**
     * @brief Loss function
     *
     * @param[in] y Predicted data
     * @param[in] t Expected data
     * @param[in] batch_size Batch size of data
     * @param[in] size Size of data
     * @return float Loss
     */
    float (*forward)(const float*, const float*, const size_t, const size_t);

    /**
     * @brief Backward of the loss
     *
     * @param[out] grad Gradient of loss function by the output
     * @param[in] y Predicted data
     * @param[in] t Expected data
     * @param[in] batch_size Batch size of data
     * @param[in] size Size of data
     */
    void (*backward)(float*, const float*, const float*, const size_t, const size_t);
} LossFunc;

#endif // LOSS_H
