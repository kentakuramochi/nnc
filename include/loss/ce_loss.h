/**
 * @file ce_loss.h
 * @brief Cross entropy loss
 *
 */
#ifndef CE_LOSS_H
#define CE_LOSS_H

#include "loss.h"

/**
 * @brief Cross entropy loss
 *
 * @return LossFunc Loss function
*/
LossFunc ce_loss(void);

#endif // CE_LOSS_H
