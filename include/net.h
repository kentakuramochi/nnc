/**
 * @file net.h
 * @brief Network structure
 *
 */
#ifndef NET_H
#define NET_H

#include "layer.h"

/**
 * @brief Sequential list of layer parameters
 * @note Terminated with NONE layer for network initialization
 *
*/
#define LAYER_PARAMS_LIST(...) (LayerParams[]){ __VA_ARGS__, (LayerParams){ .type=LAYER_TYPE_NONE } }

/**
 * @brief Network structure
 *
 */
typedef struct Net {
    int size; //!< The number of layers
    Layer *layers; //!< Layers
} Net;

/**
 * @brief Get the number of layers of a network
 *
 * @param[in] net Network
 * @return The number of layers in the network
 */
int net_size(const Net *net);

/**
 * @brief Get layers in the network
 *
 * @param[in] net Network
 * @return Pointer to layers in the network
 */
Layer *net_layers(Net *net);

/**
 * @brief Get an input layer of the network
 *
 * @param[in] net Network
 * @return Pointer to the input layer of network
 */
Layer *net_input(const Net *net);
 
/**
 * @brief Get an output layer of the network
 * @param[in] net Network
 * @return Pointer to the output layer of network
 */
Layer *net_output(const Net *net);

/**
 * @brief Allocate network layers on the heap
 *
 * @param[in,out] net Network
 * @param[in] param_list List of layer parameters
 * @return Pointer to the network, NULL if failed
 */
Net *net_alloc_layers(Net *net, LayerParams *param_list);

/**
 * @brief Free network layers allocated on the heap
 *
 * @param[in,out] net Network
 */
void net_free_layers(Net *net);

/**
 * @brief Initialize network parameters
 *
 * @param[in,out] net Network
*/
void net_init_params(Net *net);

/**
 * @brief Forward propagation of network
 *
 * @param[in,out] net Network
 * @param[in] x Network input
 * @return Pointer to the network output, NULL if failed
 */
float *net_forward(Net *net, const float *x);

/**
 * @brief Backward propagation of network
 *
 * @param[in,out] net Network
 * @param[in] dy Gradient of network output
 * @return Pointer to gradient of an input of the network, NULL if failed
 */
float *net_backward(Net *net, const float *dy);

/**
 * @brief Clear current gradients of network
 *
 * @param[in,out] net Network
 */
void net_clear_grad(Net *net);

#endif // NET_H
