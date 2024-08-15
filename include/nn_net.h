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
 * @brief Allocate network layers on the heap
 *
 * @param[in,out] net Network
 * @param[in] param_list List of layer parameters
 * @return Pointer to the network, NULL if failed
 */
NnNet *nn_net_alloc_layers(NnNet *net, NnLayerParams *param_list);

/**
 * @brief Free network layers allocated on the heap
 *
 * @param[in,out] net Network
 */
void nn_net_free_layers(NnNet *net);

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
 * @brief Clear current gradients of network
 *
 * @param[in,out] net Network
 */
void nn_net_clear_grad(NnNet *net);

#endif // NN_NET_H
