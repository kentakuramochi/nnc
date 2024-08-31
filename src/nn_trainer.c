/**
 * @file nn_trainer.c
 * @brief Train a network
 *
 */
#include "nn_trainer.h"

#include "nn_layer.h"

void nn_train_step(NnNet *net, const float learning_rate) {
    for (int i = 0; i < net->size; i++) {
        NnLayer *layer = &net->layers[i];
        NnLayerParams *params = &layer->params;

        if (layer->w != NULL) {
            for (int j = 0; j < (params->in * params->out); j++) {
                layer->w[j] -= learning_rate * layer->dw[j];
            }
        }
        if (layer->b != NULL) {
            for (int j = 0; j < params->out; j++) {
                layer->b[j] -= learning_rate * layer->db[j];
            }
        }
    }
}
