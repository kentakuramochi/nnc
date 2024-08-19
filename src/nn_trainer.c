/**
 * @file nn_trainer.c
 * @brief Train a network
 *
 */
#include "nn_trainer.h"

#include <stdlib.h>

// #include "loss.h"
#include "nn_layer.h"
// #include "nn_net.h"

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

/*void nn_net_init_random(NnNet *net) {
    for (int i = 0; i < net->size; i++) {
        NnLayer *layer = &nn_net_layers(net)[i];
        set_random_values(layer->w, (layer->in * layer->out));
        set_random_values(layer->b, layer->out);
    }
}*/

void nn_train_step(NnNet *net, const float learning_rate) {
    for (int i = 0; i < net->size; i++) {
        NnLayer *layer = &net->layers[i];
        NnLayerParams *params = &layer->params;

        for (int j = 0; j < (params->in * params->out); j++) {
            layer->w[j] -= learning_rate * layer->dw[j];
        }
        for (int j = 0; j < params->out; j++) {
            layer->b[j] -= learning_rate * layer->db[j];
        }
    }
}
