/**
 * @file scnn_net.h
 * @brief Network structure
 * 
 */
#ifndef SCNN_NET_H
#define SCNN_NET_H

#include "scnn_layer.h"

#define SCNN_NET_MAX_SIZE 256  //!< the max number of layers in network

/**
 * @brief Network structure
 * 
 */
typedef struct scnn_net {
    int         size;                       //!< the number of layers
    scnn_layer  *layers[SCNN_NET_MAX_SIZE]; //!< layers
    scnn_layer  *input;     //!< input layer
    scnn_layer  *output;    //!< output layer
} scnn_net;

/**
 * @brief Allocate network
 * 
 * @return  Pointer to network, NULL if failed
 */
scnn_net *scnn_net_alloc(void);

/**
 * @brief Append layer to network
 * 
 * @param[in,out]   net     Network
 * @param[in]       layer   Layer to be appended
 * @return                  Pointer to network, NULL if failed
 */
scnn_net *scnn_net_append(scnn_net *net, scnn_layer *layer);

/**
 * @brief Forward propagation of network
 * 
 * @param[in,out]   net Network
 * @param[in]       x   Network input
 */
void scnn_net_forward(scnn_net *net, const scnn_mat *x);

/**
 * @brief Backward propagation of network
 * 
 * @param[in,out]   net Network
 * @param[in]       t   Training label
 */
void scnn_net_backward(scnn_net *net, const scnn_mat *t);

/**
 * @brief Free network
 * 
 * @param[in,out] net   Pointer to poiner of network
 */
void scnn_net_free(scnn_net **net);

#endif // SCNN_NET_H
