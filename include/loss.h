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
 * @param[in] size Size of data
 * @return float MSE loss
 */
float mse_loss(const float *y, const float *t, const size_t size);

/**
 * @brief Calculate the squared error (SE)
 *
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
 * @return float SE loss
 */
float se_loss(const float *y, const float *t, const size_t size);

#endif // LOSS_H
