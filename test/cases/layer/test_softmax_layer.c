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
    free(layer->dx);
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
    TEST_ASSERT_NOT_NULL(layer.dx);
    TEST_ASSERT_NULL(layer.dw);
    TEST_ASSERT_NULL(layer.db);
    TEST_ASSERT_NOT_NULL(layer.forward);
    TEST_ASSERT_NOT_NULL(layer.backward);

    free_memories(&layer);
}
