/**
 * @file nn_loss.h
 * @brief Loss function
 *
 */
#ifndef NN_LOSS_H
#define NN_LOSS_H

#include <stddef.h>

/**
 * @brief Type of loss functions
*/
typedef enum NnLossType {
    NN_LOSS_TYPE_NONE, //!< None
    NN_LOSS_TYPE_BCE //!< Binary cross entropy loss
} NnLossType;

/**
 * @brief Loss function interface
 *
 * @param[in] type Type of loss function
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
 * @return float Loss
*/
float nn_loss(const int type, const float *y, const float *t, const size_t size);

/**
 * @brief Loss function backward interface
 *
 * @param[out] diff Difference of loss function by the output
 * @param[in] type Type of loss function
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
*/
void nn_loss_backward(
    float *diff, const int type, const float *y, const float *t, const size_t size
);

/**
 * @brief Table of loss functions
*/
extern float (*loss_forwards[])(const float*, const float*, const size_t);

/**
 * @brief Table of loss function backwards
*/
extern void (*loss_backwards[])(float*, const float*, const float*, const size_t);

#endif // NN_LOSS_H
