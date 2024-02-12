/**
 * @file nn_trainer.h
 * @brief Train a network
 *
 */
#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include "nn_net.h"

/**
 * @brief Train a network by one step
 *
 * @param[in,out] net Target network
 * @param[in] x Network input
 * @param[in] t Target label
 * @param[in] learning_rate Learning rate
 * @return float Loss
 */
float nn_train_step(NnNet *net, const float *x, const float *t, const float learning_rate);

#endif // NN_TRAINER_H
