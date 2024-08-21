/**
 * @file nn_loss.h
 * @brief Loss function
 *
 */
#ifndef NN_LOSS_H
#define NN_LOSS_H

#include <stddef.h>

// Temporal loss type
#define NN_LOSS_TYPE_MSE 1

/**
 * @brief Loss function interface
 *
 * @param[in] type Type of loss function
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
 * @return float Loss
*/
float nn_loss(int type, const float *y, const float *t, const size_t size);

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
 * @brief Calculate the binary cross entropy
 *
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
 * @return float Binary cross entropy loss
 */
float binary_cross_entropy_loss(const float *y, const float *t, const size_t size);

#endif // NN_LOSS_H
