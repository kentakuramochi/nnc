/**
 * @file trainer.h
 * @brief Train a network
 *
 */

#ifndef TRAINER_H
#define TRAINER_H

#include "scnn_net.h"

/**
 * @brief Train a network by one step.
 * 
 * @param[in,out] net Target network
 * @param[in] x Network input
 * @param t Target label
 * @param learning_rate Learning rate
 * @return float Loss
 */
float train_step(scnn_net *net, const float *x, const float *t, const float learning_rate);

#endif // TRAINER_H
