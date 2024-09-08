#include <stdio.h>

#include "layers.h"
#include "losses.h"
#include "trainer.h"

// Number of data
#define DATA_NUM 4

// Get a class (0/1) from the logit
static int get_binary_class(const float value) {
    return (value > 0.5f ? 1 : 0);
}

int main(void) {
    Net net;
    net_alloc_layers(
        &net,
        LAYER_PARAMS_LIST(
            { .type=LAYER_TYPE_FC, .batch_size=1, .in=2, .out=10 },
            { .type=LAYER_TYPE_SIGMOID },
            { .type=LAYER_TYPE_FC, .out=1 },
            { .type=LAYER_TYPE_SIGMOID }
        )
    );

    net_init_params(&net);

    // XOR inputs
    float x[DATA_NUM][2] = {
        { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
    };

    // XOR outputs
    float t[DATA_NUM][1] = {
        { 0 }, { 1 }, { 1 }, { 0 } 
    };

    printf("====================\n");
    printf("Start training\n");

    float grad[1];
    const int epochs = 1000;

    LossFunc loss_func = bce_loss();

    // Train with SGD
    for (int i = 0; i < epochs; i++) {
        float loss;

        for (int j = 0; j < DATA_NUM; j++) {
            net_clear_grad(&net);

            float *y = net_forward(&net, x[j]);

            loss = loss_func.forward(y, t[j], 1, 1);
            loss_func.backward(grad, y, t[j], 1, 1);

            net_backward(&net, grad);

            train_step(&net, 0.1);
        }

        if ((i + 1) % (epochs / 10) == 0) {
            printf("[%3d] loss=%f\n", (i + 1), loss);
        }
    }

    printf("Finished\n");

    printf("====================\n");
    printf("Pred./answer\n");
    printf("====================\n");
    float y[DATA_NUM];
    for (int i = 0; i < DATA_NUM; i++) {
        y[i] = net_forward(&net, x[i])[0];
        printf("%d/%d (%f)\n", (int)t[i][0], get_binary_class(y[i]), y[i]);
    }

    // Calculate the accuracy
    int corrects = 0;
    for (int i = 0; i < DATA_NUM; i++) {
        if (get_binary_class(y[i]) == (int)t[i][0]) {
            corrects++;
        }
    }

    printf("====================\n");
    printf("Accuracy=%f\n", (float)corrects / DATA_NUM);
    printf("====================\n");

    net_free_layers(&net);

    return 0;
}
