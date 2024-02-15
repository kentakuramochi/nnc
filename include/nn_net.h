/**
 * @file nn_net.h
 * @brief Network structure
 *
 */
#ifndef NN_NET_H
#define NN_NET_H

#include "nn_layer.h"

/**
 * @brief Network structure
 *
 */
typedef struct NnNet {
    int size; //!< The number of layers
    NnLayer *layers; //!< Layers
    // NnLayer *input; //!< Input layer off the network
    // NnLayer *output; //!< Output layer off the network
} NnNet;

/**
 * @brief Get the number of layers of a network
 *
 * @param[in] net Network
 * @return The number of layers in the network
 */
int nn_net_size(const NnNet *net);

/**
 * @brief Get layers in the network
 *
 * @param[in] net Network
 * @return Pointer to layers in the network
 */
NnLayer *nn_net_layers(NnNet *net);

/**
 * @brief Get an input layer of the network
 *
 * @param[in] net Network
 * @return Pointer to the input layer of network
 */
NnLayer *nn_net_input(const NnNet *net);
 
/**
 * @brief Get an output layer of the network
 * @param[in] net Network
 * @return Pointer to the output layer of network
 */
NnLayer *nn_net_output(const NnNet *net);

/**
 * @brief Allocate a network
 *
 * @return Pointer to the network, NULL if failed
 */
NnNet *nn_net_alloc(void);

/**
 * @brief Append a layer to the network
 *
 * @param[in,out] net Network
 * @param[in] params Layer parameter
 * @return Pointer to the network, NULL if failed
 */
NnNet *nn_net_append(NnNet *net, NnLayerParams params);

/**
 * @brief Initialize a network
 *
 * @param[in,out] net Network
 * @return Pointer to the network, NULL if failed
 */
NnNet *nn_net_init(NnNet *net);

/**
 * @brief Forward propagation of network
 *
 * @param[in,out] net Network
 * @param[in] x Network input
 * @return Pointer to the network output, NULL if failed
 */
float *nn_net_forward(NnNet *net, const float *x);

/**
 * @brief Backward propagation of network
 *
 * @param[in,out] net Network
 * @param[in] dy Differential of network output
 * @return Pointer to differential of an input of the network, NULL if failed
 */
float *nn_net_backward(NnNet *net, const float *dy);

/**
 * @brief Update each layer parameter
 *
 * @param[in,out] net Pointer to the network
 * @param[in] learning_rate Learning rate
 */
void nn_net_update(NnNet *net, const float learning_rate);

/**
 * @brief Free network
 *
 * @param[in,out] net Pointer to poiner of network
 */
void nn_net_free(NnNet **net);

#endif // nn_net_H
