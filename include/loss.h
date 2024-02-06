/**
 * @file loss.h
 * @brief Loss function
 *
 */

#ifndef LOSS_H
#define LOSS_H

#include <stddef.h>

/**
 * @brief Calculate the mean squared error (MSE)
 *
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param size Size of data
 * @return float MSE loss
 */
float mse_loss(const float *y, const float *t, const size_t size);

#endif // LOSS_H
