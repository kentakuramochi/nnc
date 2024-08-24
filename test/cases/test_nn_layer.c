/**
 * @file test_nn_layer.c
 * @brief Unit tests of nn_layer.c
 *
 */
#include "nn_layer.h"

#include <stdlib.h>

#include "unity.h"
#include "test_utils.h"

// Dummy layer type
#define NN_LAYER_TYPE_DUMMY 1

// Functions for the dummy layer
static float *dummy_forward(NnLayer *layer, const float *x) {
    for (int i = 0 ; i < 3; i++) {
        layer->y[i] = x[i] * 2;
    }
    return layer->y;
}

static float *dummy_backward(NnLayer *layer, const float *dy) {
    for (int i = 0 ; i < 3; i++) {
        layer->dx[i] = dy[i] / 2;
    }
    return layer->dx;
}

static NnLayer *dummy_init(NnLayer *layer) {
    layer->x = malloc(sizeof(float) * layer->params.batch_size * layer->params.in);
    layer->y = malloc(sizeof(float) * layer->params.batch_size * layer->params.out);
    layer->dx = malloc(sizeof(float) * layer->params.batch_size * layer->params.in);

    layer->forward = dummy_forward;
    layer->backward = dummy_backward;
}

NnLayer* (*nn_layer_init_funcs[NUM_LAYER_TYPE])(NnLayer*) = {
    NULL, // NN_LAYER_TYPE_NONE
    dummy_init
};

void setUp(void) {}

void tearDown(void) {}

void test_allocate_and_free(void) {
    NnLayer layer = {
        .params={
            NN_LAYER_TYPE_DUMMY, .batch_size=1, .in=2, .out=3
        }
    };

    TEST_ASSERT_EQUAL_PTR(&layer, nn_layer_alloc_params(&layer));
    TEST_ASSERT_NOT_NULL(layer.x);
    TEST_ASSERT_NOT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NOT_NULL(layer.dx);
    TEST_ASSERT_NULL(layer.dw);
    TEST_ASSERT_NULL(layer.db);
    TEST_ASSERT_EQUAL_PTR(dummy_forward, layer.forward);
    TEST_ASSERT_EQUAL_PTR(dummy_backward, layer.backward);

    nn_layer_free_params(&layer);
    TEST_ASSERT_NULL(layer.x);
    TEST_ASSERT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NULL(layer.dx);
    TEST_ASSERT_NULL(layer.dw);
    TEST_ASSERT_NULL(layer.db);
    TEST_ASSERT_NULL(layer.forward);
    TEST_ASSERT_NULL(layer.backward);
}

void test_allocation_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_alloc_params(NULL));
}

void test_allocation_fail_if_layer_type_is_not_specified(void) {
    NnLayer layer = {
        .params={ .batch_size=1, .in=2, .out=3 }
    };

    TEST_ASSERT_NULL(nn_layer_alloc_params(&layer));
    TEST_ASSERT_NULL(layer.x);
    TEST_ASSERT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NULL(layer.dx);
    TEST_ASSERT_NULL(layer.dw);
    TEST_ASSERT_NULL(layer.db);
    TEST_ASSERT_NULL(layer.forward);
    TEST_ASSERT_NULL(layer.backward);
}

void test_free_to_NULL(void) {
    nn_layer_free_params(NULL);
}

void test_connect(void) {
    NnLayer layer = {
        .params={ .batch_size=8, .in=2, .out=10 }
    };

    NnLayer next_layer = {
        .params={ .out=3 }
    };

    nn_layer_connect(&layer, &next_layer);

    TEST_ASSERT_EQUAL_INT(8, next_layer.params.batch_size);
    TEST_ASSERT_EQUAL_INT(10, next_layer.params.in);
}

void test_forward(void) {
    NnLayer layer = {
        .y = TEST_UTIL_FLOAT_ZEROS(3),
        .forward = dummy_forward
    };

    float answer[] = { -2, 0, 2 };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        answer, nn_layer_forward(&layer, TEST_UTIL_FLOAT_ARRAY(-1, 0, 1)), 3
    );
}

void test_forward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_forward(NULL, TEST_UTIL_FLOAT_ZEROS(1)));
}

void test_forward_fail_if_x_is_NULL(void) {
    NnLayer layer = {
        .y = TEST_UTIL_FLOAT_ZEROS(3),
        .forward = dummy_forward
    };

    TEST_ASSERT_NULL(nn_layer_forward(&layer, NULL));
}

void test_backward(void) {
    NnLayer layer = {
        .dx = TEST_UTIL_FLOAT_ZEROS(3),
        .backward = dummy_backward
    };

    float answer[] = { -1, 0, 1 };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        answer, nn_layer_backward(&layer, TEST_UTIL_FLOAT_ARRAY(-2, 0, 2)), 3
    );
}

void test_backward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_backward(NULL, TEST_UTIL_FLOAT_ZEROS(1)));
}

void test_backward_fail_if_dy_is_NULL(void) {
    NnLayer layer = {
        .dx = TEST_UTIL_FLOAT_ZEROS(3),
        .backward = dummy_backward
    };

    TEST_ASSERT_NULL(nn_layer_backward(&layer, NULL));
}

void test_clear_grad(void) {
    NnLayer layer = {
        .params = { .batch_size=4, .in=2, .out=3 },
        .dx = TEST_UTIL_FLOAT_ARRAY(-1, 1, -2, 2, -3, 3, -4, 4),
        .dw = TEST_UTIL_FLOAT_ARRAY(1, 2, 3, 4, 5, 6),
        .db = TEST_UTIL_FLOAT_ARRAY(-1, -2, -3),
    };

    nn_layer_clear_grad(&layer);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ZEROS(2), layer.dx, 2
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ZEROS(3 * 2), layer.dw, (3 * 2)
    );
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        TEST_UTIL_FLOAT_ZEROS(3), layer.db, 3
    );
}
