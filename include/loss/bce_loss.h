/**
 * @file bce_loss.h
 * @brief Binary cross entropy loss
 *
 */
#ifndef BCE_LOSS_H
#define BCE_LOSS_H

#include "nn_loss.h"

// Temporal parameter for the loss type
#define NN_LOSS_TYPE_BCE 1

/**
 * @brief Binary cross entropy loss
 *
 * @param[in] y Predicted data
 * @param[in] t Expected data
 * @param[in] size Size of data
 * @return float Loss
*/
float bce_loss(const float *y, const float *t, const size_t size);

#endif // BCE_LOSS_H
