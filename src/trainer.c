/**
 * @file trainer.c
 * @brief Train a network
 *
 */
#include "trainer.h"

#include "layer.h"

void train_step(Net *net, const float learning_rate) {
    for (int i = 0; i < net->size; i++) {
        Layer *layer = &net->layers[i];
        LayerParams *params = &layer->params;

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
