/**
 * @file test_trainer.c
 * @brief Unit tests of trainer.c
 *
 */

#include "trainer.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "activation.h"
#include "loss.h"
#include "scnn_blas.h"
#include "scnn_layer.h"
#include "scnn_net.h"

#include "unity.h"

void setUp(void)
{
}

void tearDown(void)
{
}

void init_random(float *x, const size_t size)
{
    for (size_t i = 0; i < size; i++) {
        x[i] = (float)rand() / RAND_MAX - 0.5f;
    }
}

void test_train_step(void)
{
    scnn_net net = {
        .size = 2,
        .batch_size = 1,
        .layers = (scnn_layer[]){
            {
                .in = 2,
                .out = 3,
                .x = (float[2]){ 0 },
                .y = (float[3]){ 0 },
                .z = (float[3]){ 0 },
                .w = (float[3 * 2]){ 0 },
                .b = (float[3]){ 0 },
                .dx = (float[2]){ 0 },
                .dz = (float[3]){ 0 },
                .dw = (float[3 * 2]){ 0 },
                .db = (float[3]){ 0 },
            },
            {
                .in = 3,
                .out = 1,
                .x = (float[3]){ 0 },
                .y = (float[1]){ 0 },
                .z = (float[1]){ 0 },
                .w = (float[1 * 3]){ 0 },
                .b = (float[1]){ 0 },
                .dx = (float[3]){ 0 },
                .dz = (float[1]){ 0 },
                .dw = (float[1 * 3]){ 0 },
                .db = (float[1]){ 0 },
            }
        }
    };

    init_random(net.layers[0].w, (3 * 2));
    init_random(net.layers[0].b, 3);
    init_random(net.layers[1].w, (1 * 3));
    init_random(net.layers[1].b, 1);

    float x[] = {
        0.1, 0.1
    };

    float t[] = {
        1
    };

    float loss = 0;
    float prev_loss = FLT_MAX;
    for (int i = 0; i < 10; i++) {
        loss = train_step(&net, x, t, 0.1);

        TEST_ASSERT_TRUE(prev_loss > loss);

        prev_loss = loss;
    }
}
