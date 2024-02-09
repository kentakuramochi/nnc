/**
 * @file trainer.c
 * @brief Train a network
 *
 */

#include "trainer.h"

#include <stdlib.h>

#include "scnn_net.h"
#include "scnn_layer.h"
#include "loss.h"

float train_step(scnn_net *net, const float *x, const float *t, const float learning_rate)
{
    const float *y = scnn_net_forward(net, x);

    const int osize = scnn_net_layers(net)[net->size - 1].params.out;

    float *dy = malloc(sizeof(float) * osize);

    // Get difference between the output and the label
    for (int i = 0; i < osize; i++) {
        dy[i] = y[i] - t[i];
    }

    float loss = mse_loss(y, t, osize);

    scnn_net_backward(net, dy);

    net_update(net, learning_rate);

    free(dy);
    dy = NULL;

    return loss;
}
