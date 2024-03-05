/**
 * @file nn_trainer.h
 * @brief Train a network
 *
 */
#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include <stddef.h>

#include "nn_net.h"

/**
 * @brief Initialize network parameters with random values
 * 
 * @param[in,out] net Target network
*/
void nn_net_init_random(NnNet *net);

/**
 * @brief Train a network by one step
 *
 * @param[in,out] net Target network
 * @param[in] x Network input
 * @param[in] t Target label
 * @param[in] learning_rate Learning rate
 * @param[in] loss_func Loss function
 * @return float Loss
 */
float nn_train_step(
    NnNet *net, const float *x, const float *t, const float learning_rate,
    float (*loss_func)(const float*, const float*, const size_t)
);

#endif // NN_TRAINER_H
