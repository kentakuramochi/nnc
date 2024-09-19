/**
 * @file bce_loss.h
 * @brief Binary cross entropy loss
 */
#ifndef BCE_LOSS_H
#define BCE_LOSS_H

#include "loss.h"

/**
 * @brief Binary cross entropy loss
 *
 * @return LossFunc Loss function
 */
LossFunc bce_loss(void);

#endif // BCE_LOSS_H
