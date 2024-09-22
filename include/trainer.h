/**
 * @file trainer.h
 * @brief Train a network
 */
#ifndef TRAINER_H
#define TRAINER_H

#include "net.h"

/**
 * @brief Train a network in one step
 *
 * @param[in,out] net Target network
 * @param[in] learning_rate Learning rate
 */
void train_step(Net *net, const float learning_rate);

#endif // TRAINER_H
