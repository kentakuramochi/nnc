/**
 * @file test_nn_layer.c
 * @brief Unit tests of nn_layer.c
 *
 */
#include "nn_layer.h"

#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_allocate_and_free(void) {
    NnLayer layer = {
        .params={
            NN_LAYER_TYPE_IDENTITY, .batch_size=1, .in=2, .out=3
        }
    };

    // Test with an identity layer
    TEST_ASSERT_EQUAL_PTR(&layer, nn_layer_alloc_params(&layer));
    TEST_ASSERT_NOT_NULL(layer.x);
    TEST_ASSERT_NOT_NULL(layer.y);
    TEST_ASSERT_NULL(layer.w);
    TEST_ASSERT_NULL(layer.b);
    TEST_ASSERT_NOT_NULL(layer.dx);
    TEST_ASSERT_NULL(layer.dw);
    TEST_ASSERT_NULL(layer.db);
    TEST_ASSERT_EQUAL_PTR(identity_forward, layer.forward);
    TEST_ASSERT_EQUAL_PTR(identity_backward, layer.backward);

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

void test_allocation_fail_if_parameters_contains_0(void) {
    NnLayer layer[3] = {
        { .params={ NN_LAYER_TYPE_IDENTITY, .batch_size=0, .in=2, .out=3 } },
        { .params={ NN_LAYER_TYPE_IDENTITY, .batch_size=1, .in=0, .out=3 } },
        { .params={ NN_LAYER_TYPE_IDENTITY, .batch_size=1, .in=2, .out=0 } }
    };

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_NULL(nn_layer_alloc_params(&layer[i]));
        TEST_ASSERT_NULL(layer[i].x);
        TEST_ASSERT_NULL(layer[i].y);
        TEST_ASSERT_NULL(layer[i].w);
        TEST_ASSERT_NULL(layer[i].b);
        TEST_ASSERT_NULL(layer[i].dx);
        TEST_ASSERT_NULL(layer[i].dw);
        TEST_ASSERT_NULL(layer[i].db);
        TEST_ASSERT_NULL(layer[i].forward);
        TEST_ASSERT_NULL(layer[i].backward);
    }
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

float *dummy_forward(NnLayer *layer, const float *x) {
    for (int i = 0 ; i < 3; i++) {
        layer->y[i] = x[i] * 2;
    }
    return layer->y;
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

float *dummy_backward(NnLayer *layer, const float *dy) {
    for (int i = 0 ; i < 3; i++) {
        layer->dx[i] = dy[i] / 2;
    }
    return layer->dx;
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
