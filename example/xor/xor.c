#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "loss.h"
#include "nn_trainer.h"

#define BATCH_SIZE 4

static int get_class(const float value) {
    return (value > 0.5f ? 1 : 0);
}

static float get_accuracy(const float *y, const float *t, const size_t size) {
    int correct = 0;

    for (size_t i = 0; i < size; i++) {
        if (get_class(y[i]) == (int)t[i]) {
            correct++;
        }
    }

    return (float)correct / size;
}

int main(void) {
    NnNet *net = nn_net_alloc();
    nn_net_append(net, (NnLayerParams){ .batch_size=BATCH_SIZE, .in=2, .out=100 });
    nn_net_append(net, (NnLayerParams){ .out=10 });
    nn_net_append(net, (NnLayerParams){ .out=1 });

    nn_net_init(net);

    nn_net_init_random(net);

    float x[][BATCH_SIZE * 2] = {
        {
            0.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f
        }
    };

    float t[][BATCH_SIZE * 1] = {
        {
            0,
            1,
            1,
            0
        } 
    };

    printf("********************\n");
    printf("Start learning\n");
    printf("********************\n");

    const int iter = 1000;
    for (int i = 0; i < iter; i++) {
        float loss = nn_train_step(net, x[0], t[0], 0.5, se_loss);
        if ((i + 1) % (iter / 10) == 0) {
            printf("[%4d] loss=%f\n", (i + 1), loss);
        }
    }

    printf("********************\n");
    printf("Answer/predict\n");
    printf("********************\n");
    float *y = nn_net_forward(net, x[0]);
    for (int i = 0; i < BATCH_SIZE; i++) {
        printf("%d/%d (%f)\n", (int)t[0][i], get_class(y[i]), y[i]);
    }

    printf("********************\n");
    printf("Accuracy=%f\n", get_accuracy(y, t[0], BATCH_SIZE));
    printf("********************\n");

    nn_net_free(&net);

    return 0;
}
