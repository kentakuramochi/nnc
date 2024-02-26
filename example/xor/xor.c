#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "loss.h"
#include "nn_trainer.h"

void init_rand(float *array, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = 2 * ((float)rand() / RAND_MAX) - 1;
    }
}

void net_randomize(NnNet *net) {
    srand(time(NULL));

    for (int i = 0; i < net->size; i++) {
        NnLayer *layer = &nn_net_layers(net)[i];
        init_rand(layer->w, (layer->in * layer->out));
        init_rand(layer->b, layer->out);
    }
}

int main(void) {
    NnNet *net = nn_net_alloc();
    nn_net_append(net, (NnLayerParams){ .batch_size=4, .in=2, .out=100 });
    nn_net_append(net, (NnLayerParams){ .out=10 });
    nn_net_append(net, (NnLayerParams){ .out=1 });

    nn_net_init(net);

    net_randomize(net);

    float x[][4 * 2] = {
        {
            0.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f
        }
    };

    float t[][4 * 1] = {
        {
            0,
            1,
            1,
            0
        } 
    };

    const int iter = 100;
    for (int i = 0; i < iter; i++) {
        float loss = nn_train_step(net, x[0], t[0], 0.01);
        if ((i + 1) % (iter / 10) == 0) {
            printf("[%4d] loss=%f\n", (i + 1), loss);
        }
    }

    float *y = nn_net_forward(net, x[0]);
    for (int i = 0; i < 4; i++) {
        printf("%f (%d)\n", y[i], (y[i] > 0.5f ? 1 : 0));
    }

    nn_net_free(&net);

    return 0;
}
