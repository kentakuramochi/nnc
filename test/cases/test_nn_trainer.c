/**
 * @file test_nn_trainer.c
 * @brief Unit tests of trainer.c
 *
 */
#include "nn_trainer.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "activation.h"
#include "blas.h"
#include "loss.h"
#include "nn_layer.h"
#include "nn_net.h"

#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_train_step(void) {
    NnNet net = {
        .size = 2,
        .layers = (NnLayer[]){
            {
                .batch_size = 1,
                .in = 2,
                .out = 3,
                .x = FLOAT_ZEROS(2),
                .y = FLOAT_ZEROS(3),
                .z = FLOAT_ZEROS(3),
                .w = FLOAT_ZEROS(3 * 2),
                .b = FLOAT_ZEROS(3),
                .dx = FLOAT_ZEROS(2),
                .dz = FLOAT_ZEROS(3),
                .dw = FLOAT_ZEROS(3 * 2),
                .db = FLOAT_ZEROS(3)
            },
            {
                .batch_size = 1,
                .in = 3,
                .out = 1,
                .x = FLOAT_ZEROS(3),
                .y = FLOAT_ZEROS(1),
                .z = FLOAT_ZEROS(1),
                .w = FLOAT_ZEROS(1 * 3),
                .b = FLOAT_ZEROS(1),
                .dx = FLOAT_ZEROS(3),
                .dz = FLOAT_ZEROS(1),
                .dw = FLOAT_ZEROS(1 * 3),
                .db = FLOAT_ZEROS(1)
            }
        }
    };

    nn_net_init_random(&net);

    float x[] = {
        0.1, 0.1
    };

    float t[] = {
        1
    };

    float loss = 0;
    float prev_loss = FLT_MAX;
    for (int i = 0; i < 10; i++) {
        loss = nn_train_step(&net, x, t, 0.1, se_loss);

        TEST_ASSERT_TRUE(prev_loss > loss);

        prev_loss = loss;
    }
}
