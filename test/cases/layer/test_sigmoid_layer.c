/**
 * @file test_sigmoid_layer.c
 * @brief Unit tests of sigmoid_layer.c
 *
 */
#include "sigmoid_layer.h"

#include <stdlib.h>

#include "mock_nn_layer.h"
#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

static void free_memories(NnLayer *layer) {
    free(layer->x);
    free(layer->y);
    free(layer->dx);
}

void test_alloc_and_free(void) {
    NnLayer layer = {
        .params={ NN_LAYER_TYPE_SIGMOID, .batch_size=1, .in=2 }
    };

    TEST_ASSERT_EQUAL_PTR(&layer, sigmoid_layer_init(&layer));
    TEST_ASSERT_EQUAL_INT(2, layer.params.out);
    TEST_ASSERT_NOT_NULL(layer.x);
    TEST_ASSERT_NOT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NOT_NULL(layer.dx);
    TEST_ASSERT_NULL(layer.dw);
    TEST_ASSERT_NULL(layer.db);
    TEST_ASSERT_NOT_NULL(layer.forward);
    TEST_ASSERT_NOT_NULL(layer.backward);

    free_memories(&layer);
}

void test_forward(void) {
    NnLayer layer = {
        .params={ NN_LAYER_TYPE_SIGMOID, .batch_size=2, .in=3 }
    };

    sigmoid_layer_init(&layer);

    float x[] = {
        -1, -0.5, -0.25,
        0.25, 0.5, 1
    };

    float y[] = {
        0.268941, 0.377541, 0.437823,
        0.562177, 0.622459, 0.731059
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        y, layer.forward(&layer, x), (2 * 3)
    );

    free_memories(&layer);
}

void test_backward(void) {
    NnLayer layer = {
        .params={ NN_LAYER_TYPE_SIGMOID, .batch_size=2, .in=3 }
    };

    sigmoid_layer_init(&layer);

    test_util_copy_array(
        layer.y,
        TEST_UTIL_FLOAT_ARRAY(
            0.268941, 0.377541, 0.437823,
            0.562177, 0.622459, 0.731059
        ),
        (sizeof(float) * (2 * 3))
    );

    float dy[] = {
        -3, -2, -1,
        1, 2, 3
    };

    float dx[] = {
        -0.589836, -0.470008, -0.246134,
        0.246134, 0.470008, 0.589836
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        dx, layer.backward(&layer, dy), (2 * 3)
    );

    free_memories(&layer);
}
