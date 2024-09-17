/**
 * @file test_softmax_layer.c
 * @brief Unit tests of softmax_layer.c
 *
 */
#include "softmax_layer.h"

#include <stdlib.h>

#include "mock_layer.h"
#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

static void free_memories(Layer *layer) {
    free(layer->x);
    free(layer->y);
    free(layer->gx);
}

void test_alloc_and_free(void) {
    Layer layer = {
        .params={ LAYER_TYPE_SOFTMAX, .batch_size=1, .in=3 }
    };

    TEST_ASSERT_EQUAL_PTR(&layer, softmax_layer_init(&layer));
    TEST_ASSERT_EQUAL_INT(3, layer.params.out);
    TEST_ASSERT_NOT_NULL(layer.x);
    TEST_ASSERT_NOT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NOT_NULL(layer.gx);
    TEST_ASSERT_NULL(layer.gw);
    TEST_ASSERT_NULL(layer.gb);
    TEST_ASSERT_NOT_NULL(layer.forward);
    TEST_ASSERT_NOT_NULL(layer.backward);

    free_memories(&layer);
}

void test_forward(void) {
    Layer layer = {
        .params={ LAYER_TYPE_SOFTMAX, .batch_size=2, .in=3 }
    };

    softmax_layer_init(&layer);

    float x[] = {
        0.8, 0.2, 0.1,
        0.3, -0.9, 0.4
    };

    float y[] = {
        0.488903, 0.268315, 0.242782,
        0.415565, 0.125166, 0.459270
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        y, layer.forward(&layer, x), (2 * 3)
    );

    free_memories(&layer);
}

void test_backward(void) {
    Layer layer = {
        .params={ LAYER_TYPE_SOFTMAX, .batch_size=2, .in=3 }
    };

    softmax_layer_init(&layer);

    test_util_copy_array(
        layer.y,
        TEST_UTIL_FLOAT_ARRAY(
            0.488903, 0.268315, 0.242782,
            0.415565, 0.125166, 0.459270
        ),
        (sizeof(float) * (2 * 3))
    );

    float dy[] = {
        -0.5110970, 0.2683150, 0.2427820,
        0.4155650, 0.1251660, -0.5407300
    };

    float gx[] = {
        -0.1917263, 0.1039066, 0.0878197,
        0.1976198, 0.0231740, -0.2207938
    };

    // Checking a result with considering about accuracy is difficult...
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        gx, layer.backward(&layer, dy), (2 * 3)
    );

    free_memories(&layer);
}
