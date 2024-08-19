/**
 * @file nn_trainer.h
 * @brief Train a network
 *
 */
#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include "nn_net.h"

/**
 * @brief Initialize network parameters with random values
 * 
 * @param[in,out] net Target network
*/
void nn_net_init_random(NnNet *net);

/**
 * @brief Train a network in one step
 *
 * @param[in,out] net Target network
 * @param[in] learning_rate Learning rate
 */
void nn_train_step(NnNet *net, const float learning_rate);

#endif // NN_TRAINER_H
