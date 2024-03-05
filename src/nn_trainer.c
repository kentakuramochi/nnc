/**
 * @file nn_trainer.c
 * @brief Train a network
 *
 */
#include "nn_trainer.h"

#include <stdlib.h>

#include "loss.h"
#include "nn_layer.h"
#include "nn_net.h"

/**
 * @brief Set random values to an array
 * 
 * @param[in,out] array Array to set random values
 * @param[in] size Size of the array
*/
static void set_random_values(float *array, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = 2 * ((float)rand() / RAND_MAX) - 1;
    }
}

void nn_net_init_random(NnNet *net) {
    for (int i = 0; i < net->size; i++) {
        NnLayer *layer = &nn_net_layers(net)[i];
        set_random_values(layer->w, (layer->in * layer->out));
        set_random_values(layer->b, layer->out);
    }
}

float nn_train_step(
    NnNet *net, const float *x, const float *t, const float learning_rate,
    float (*loss_func)(const float*, const float*, const size_t)
) {
    const float *y = nn_net_forward(net, x);

    NnLayer *layer = &nn_net_layers(net)[net->size - 1];
    const int osize = layer->batch_size * layer->out;

    float *dy = malloc(sizeof(float) * osize);

    // Get difference between the output and the label
    for (int i = 0; i < osize; i++) {
        dy[i] = y[i] - t[i];
    }

    float loss = loss_func(y, t, osize);

    nn_net_backward(net, dy);

    nn_net_update(net, learning_rate);

    free(dy);
    dy = NULL;

    return loss;
}
