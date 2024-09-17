/**
 * @file test_layer.c
 * @brief Unit tests of layer.c
 *
 */
#include "layer.h"

#include <stdlib.h>

#include "unity.h"
#include "test_utils.h"

// Dummy layer type
#define LAYER_TYPE_DUMMY 1

// Functions for the dummy layer
static float *dummy_forward(Layer *layer, const float *x) {
    for (int i = 0 ; i < (layer->params.batch_size * layer->params.in); i++) {
        layer->y[i] = x[i] * 2;
    }
    return layer->y;
}

static float *dummy_backward(Layer *layer, const float *gy) {
    for (int i = 0 ; i < (layer->params.batch_size * layer->params.out); i++) {
        layer->gx[i] = gy[i] / 2;
    }
    return layer->gx;
}

static Layer *dummy_init(Layer *layer) {
    layer->x = malloc(sizeof(float) * layer->params.batch_size * layer->params.in);
    layer->y = malloc(sizeof(float) * layer->params.batch_size * layer->params.out);
    layer->gx = malloc(sizeof(float) * layer->params.batch_size * layer->params.in);

    layer->forward = dummy_forward;
    layer->backward = dummy_backward;
}

Layer* (*layer_init_funcs[])(Layer*) = {
    NULL, // LAYER_TYPE_NONE
    dummy_init
};

void setUp(void) {}

void tearDown(void) {}

void test_allocate_and_free(void) {
    Layer layer = {
        .params={ LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=2 }
    };

    TEST_ASSERT_EQUAL_PTR(&layer, layer_alloc_params(&layer));
    TEST_ASSERT_NOT_NULL(layer.x);
    TEST_ASSERT_NOT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NOT_NULL(layer.gx);
    TEST_ASSERT_NULL(layer.gw);
    TEST_ASSERT_NULL(layer.gb);
    TEST_ASSERT_EQUAL_PTR(dummy_forward, layer.forward);
    TEST_ASSERT_EQUAL_PTR(dummy_backward, layer.backward);

    layer_free_params(&layer);
    TEST_ASSERT_NULL(layer.x);
    TEST_ASSERT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NULL(layer.gx);
    TEST_ASSERT_NULL(layer.gw);
    TEST_ASSERT_NULL(layer.gb);
    TEST_ASSERT_NULL(layer.forward);
    TEST_ASSERT_NULL(layer.backward);
}

void test_allocation_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(layer_alloc_params(NULL));
}

void test_allocation_fail_if_layer_type_is_not_specified(void) {
    Layer layer = { .params={ .batch_size=1, .in=2, .out=2 } };

    TEST_ASSERT_NULL(layer_alloc_params(&layer));
    TEST_ASSERT_NULL(layer.x);
    TEST_ASSERT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NULL(layer.gx);
    TEST_ASSERT_NULL(layer.gw);
    TEST_ASSERT_NULL(layer.gb);
    TEST_ASSERT_NULL(layer.forward);
    TEST_ASSERT_NULL(layer.backward);
}

void test_free_to_NULL(void) {
    layer_free_params(NULL);
}

void test_connect(void) {
    Layer layer = { .params={ .batch_size=8, .in=2, .out=10 } };
    Layer next_layer = { .params={ .out=3 } };

    TEST_ASSERT_TRUE(layer_connect(&layer, &next_layer));

    TEST_ASSERT_EQUAL_INT(8, next_layer.params.batch_size);
    TEST_ASSERT_EQUAL_INT(10, next_layer.params.in);
}

void test_fail_connect_when_different_value_is_set(void) {
    Layer layer = { .params={ .batch_size=8, .in=2, .out=10 } };
    Layer next_layer = { .params={ .in=5, .out=3 } };

    TEST_ASSERT_FALSE(layer_connect(&layer, &next_layer));

    TEST_ASSERT_EQUAL_INT(0, next_layer.params.batch_size);
    TEST_ASSERT_EQUAL_INT(5, next_layer.params.in);
}

void test_forward(void) {
    Layer layer = {
        .params = { .batch_size=1, .in=3, .out=3 },
        .y = TEST_UTIL_FLOAT_ZEROS(3),
        .forward = dummy_forward
    };

    float answer[] = { -2, 0, 2 };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        answer, layer_forward(&layer, TEST_UTIL_FLOAT_ARRAY(-1, 0, 1)), 3
    );
}

void test_forward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(layer_forward(NULL, TEST_UTIL_FLOAT_ZEROS(1)));
}

void test_forward_fail_if_x_is_NULL(void) {
    Layer layer = {
        .params = { .batch_size=1, .in=3, .out=3 },
        .y = TEST_UTIL_FLOAT_ZEROS(3),
        .forward = dummy_forward
    };

    TEST_ASSERT_NULL(layer_forward(&layer, NULL));
}

void test_backward(void) {
    Layer layer = {
        .params = { .batch_size=1, .in=3, .out=3 },
        .gx = TEST_UTIL_FLOAT_ZEROS(3),
        .backward = dummy_backward
    };

    float answer[] = { -1, 0, 1 };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        answer, layer_backward(&layer, TEST_UTIL_FLOAT_ARRAY(-2, 0, 2)), 3
    );
}

void test_backward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(layer_backward(NULL, TEST_UTIL_FLOAT_ZEROS(1)));
}

void test_backward_fail_if_dy_is_NULL(void) {
    Layer layer = {
        .params = { .batch_size=1, .in=3, .out=3 },
        .gx = TEST_UTIL_FLOAT_ZEROS(3),
        .backward = dummy_backward
    };

    TEST_ASSERT_NULL(layer_backward(&layer, NULL));
}

void test_clear_grad(void) {
    Layer layers[] = {
        {
            .params = { .batch_size=4, .in=2, .out=3 },
            .gx = TEST_UTIL_FLOAT_ARRAY(-1, 1, -2, 2, -3, 3, -4, 4),
            .gw = TEST_UTIL_FLOAT_ARRAY(1, 2, 3, 4, 5, 6),
            .gb = TEST_UTIL_FLOAT_ARRAY(-1, -2, -3),
        },
        { .params = { .batch_size=4, .in=3, .out=1 } }
    };

    layer_clear_grad(&layers[0]);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ZEROS(2), layers[0].gx, 2
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ZEROS(3 * 2), layers[0].gw, (3 * 2)
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ZEROS(3), layers[0].gb, 3
    );

    // Skip unallocated grads
    layer_clear_grad(&layers[1]);
}
