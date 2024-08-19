/**
 * @file test_nn_trainer.c
 * @brief Unit tests of nn_trainer.c
 *
 */
#include "nn_trainer.h"

#include "mock_nn_net.h"
#include "test_utils.h"
#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void test_train_step(void) {
    NnLayer layers[] = {
        {
            .params={ .batch_size=1, .in=2, .out=3 },
            // 3x2, thought fully connected layer
            .w=TEST_UTIL_FLOAT_ARRAY(1, 1, 1, 1, 1, 1),
            .dw=TEST_UTIL_FLOAT_ARRAY(1, 2, 3, 4, 5, 6),
            .b=TEST_UTIL_FLOAT_ARRAY(1, 1, 1),
            .db=TEST_UTIL_FLOAT_ARRAY(0, 1, 2)
        },
        {
            .params={ .batch_size=1, .in=3, .out=1 },
            // 1x3
            .w=TEST_UTIL_FLOAT_ARRAY(1, 1, 1),
            .dw=TEST_UTIL_FLOAT_ARRAY(-2, -1, 0),
            .b=TEST_UTIL_FLOAT_ARRAY(1),
            .db=TEST_UTIL_FLOAT_ARRAY(-1)
        }
    };

    NnNet net = {
        .size = 2,
        .layers = layers
    };

    nn_train_step(&net, 0.01);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ARRAY(0.99, 0.98, 0.97, 0.96, 0.95, 0.94),
        net.layers[0].w,
        (3 * 2)
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ARRAY(1, 0.99, 0.98), net.layers[0].b, 3
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ARRAY(1.02, 1.01, 1), net.layers[1].w, (3 * 1)
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ARRAY(1.01), net.layers[1].b, 1
    );
}
