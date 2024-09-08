/**
 * @file test_fc_layer.c
 * @brief Unit tests of fc_layer.c
 *
 */
#include "fc_layer.h"

#include <stdlib.h>

#include "mock_layer.h"
#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

static void free_memories(Layer *layer) {
    free(layer->x);
    free(layer->y);
    free(layer->w);
    free(layer->b);
    free(layer->dx);
    free(layer->dw);
    free(layer->db);
}

void test_alloc_and_free(void) {
    Layer layer = {
        .params={ LAYER_TYPE_FC, .batch_size=1, .in=2, .out=3 }
    };

    TEST_ASSERT_EQUAL_PTR(&layer, fc_layer_init(&layer));
    TEST_ASSERT_NOT_NULL(layer.x);
    TEST_ASSERT_NOT_NULL(layer.y);
    TEST_ASSERT_NOT_NULL(layer.w);
    TEST_ASSERT_NOT_NULL(layer.b);
    TEST_ASSERT_NOT_NULL(layer.dx);
    TEST_ASSERT_NOT_NULL(layer.dw);
    TEST_ASSERT_NOT_NULL(layer.db);
    TEST_ASSERT_NOT_NULL(layer.forward);
    TEST_ASSERT_NOT_NULL(layer.backward);

    free_memories(&layer);
}

void test_forward(void) {
    Layer layer = {
        .params={ LAYER_TYPE_FC, .batch_size=2, .in=2, .out=3 }
    };

    fc_layer_init(&layer);

    test_util_copy_array(
        layer.w,
        TEST_UTIL_FLOAT_ARRAY(
            0, 1,
            0, -1,
            1, 1,
        ),
        (sizeof(float) * (3 * 2))
    );

    test_util_copy_array(
        layer.b, TEST_UTIL_FLOAT_ARRAY(-1, 0, 1), (sizeof(float) * 3)
    );

    float x[] = {
        1, 1,
        -1, -1
    };

    float y[] = {
        0, -1, 3,
        -2, 1, -1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        y, layer.forward(&layer, x), (2 * 3)
    );

    free_memories(&layer);
}

void test_backward(void) {
    Layer layer = {
        .params={ LAYER_TYPE_FC, .batch_size=2, .in=2, .out=3 }
    };

    fc_layer_init(&layer);

    test_util_copy_array(
        layer.x,
        TEST_UTIL_FLOAT_ARRAY(
            1, 1,
            -1, -1
        ),
        (sizeof(float) * (2 * 2))
    );

    test_util_copy_array(
        layer.w,
        TEST_UTIL_FLOAT_ARRAY(
            0, 1,
            0, -1,
            1, 1,
        ),
        (sizeof(float) * (3 * 2))
    );

    // Clear gradients
    test_util_copy_array(
        layer.dx, TEST_UTIL_FLOAT_ZEROS(2 * 2), (sizeof(float) * (2 * 2))
    );

    test_util_copy_array(
        layer.dw, TEST_UTIL_FLOAT_ZEROS(3 * 2), (sizeof(float) * (3 * 2))
    );

    test_util_copy_array(
        layer.db, TEST_UTIL_FLOAT_ZEROS(3), (sizeof(float) * 3)
    );

    float dy[] = {
        0, 1, -3,
        2, -1, 1
    };

    float dx[] = {
        -3, -4,
        1, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        dx, layer.backward(&layer, dy), (2 * 2)
    );

    float dw[] = {
        -2, -2,
        2, 2,
        -4, -4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dw, layer.dw, (3 * 2));

    float db[] = {
        2, 0, -2
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(db, layer.db, 3);

    free_memories(&layer);
}
