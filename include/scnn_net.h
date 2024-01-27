/**
 * @file scnn_net.h
 * @brief Network structure
 * 
 */
#ifndef SCNN_NET_H
#define SCNN_NET_H

#include "scnn_layer.h"

/**
 * @brief Network structure
 * 
 */
typedef struct scnn_net {
    int         size;       //!< The number of layers
    int         batch_size; //!< Batch size
    scnn_layer  *layers;    //!< Layers
    scnn_layer  *input;     //!< Input layer off the network
    scnn_layer  *output;    //!< Output layer off the network
} scnn_net;

/**
 * @brief Get the number of layers of a network
 * 
 * @param[in]   net Network
 * @return          The number of layers in the network
*/
int scnn_net_size(const scnn_net *net);

/**
 * @brief Get batch size of a network
 * 
 * @param[in]   net Network
 * @return          Batch size of the network
*/
int scnn_net_batch_size(const scnn_net *net);

/**
 * @brief Get layers in the network
 * 
 * @param[in]   net Network
 * @return          Pointer to layers in the network
*/
scnn_layer *scnn_net_layers(scnn_net *net);

/**
 * @brief Get an input layer of the network
 *
 * @param[in]   net Network
 * @return          Pointer to the input layer of network
*/
scnn_layer *scnn_net_input(const scnn_net *net);
 
/**
 * @brief Get an output layer of the network
 * @param[in]   net Network
 * @return          Pointer to the output layer of network
*/
scnn_layer *scnn_net_output(const scnn_net *net);

/**
 * @brief Allocate a network
 * 
 * @return  Pointer to the network, NULL if failed
 */
scnn_net *scnn_net_alloc(void);

/**
 * @brief Append a layer to the network
 * 
 * @param[in,out]   net     Network
 * @param[in]       params  Layer parameter
 * @return                  Pointer to the network, NULL if failed
 */
scnn_net *scnn_net_append(scnn_net *net, scnn_layer_params params);

/**
 * @brief Initialize a network
 * 
 * @param[in,out] net   Network
 * @return              Pointer to the network, NULL if failed
 */
scnn_net *scnn_net_init(scnn_net *net);

/**
 * @brief Forward propagation of network
 * 
 * @param[in,out]   net Network
 * @param[in]       x   Network input
 * @return              Pointer to the network output, NULL if failed
 */
scnn_dtype *scnn_net_forward(scnn_net *net, const scnn_dtype *x);

/**
 * @brief Backward propagation of network
 * 
 * @param[in,out]   net Network
 * @param[in]       dy  Differential of network output
 * @return              Pointer to differential of an input of the network, NULL if failed
 */
scnn_dtype *scnn_net_backward(scnn_net *net, const scnn_dtype *dy);

/**
 * @brief Free network
 * 
 * @param[in,out] net   Pointer to poiner of network
 */
void scnn_net_free(scnn_net **net);

#endif // SCNN_NET_H
