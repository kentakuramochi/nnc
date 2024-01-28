/**
 * @file test_scnn_net.c
 * @brief Unit tests of scnn_net.c
 * 
 */
#include "scnn_net.h"

#include "unity.h"

#include "mock_scnn_layer.h"

#define INIT_LAYER_SIZE 128

scnn_net *net;

scnn_layer dummy_layer;
scnn_layer dummy_layers[INIT_LAYER_SIZE];

scnn_layer_params param = {
    SCNN_LAYER_FC, .in_shape = { 1, 3, 28, 28 }, .out = 10
};

scnn_layer_params params3[] = {
    { SCNN_LAYER_FC, .in_shape = { 1, 3, 28, 28 }, .out = 100 },
    { SCNN_LAYER_FC, .out = 10 },
    { SCNN_LAYER_SIGMOID },
};

void setUp(void)
{
    net = NULL;
}

void tearDown(void)
{}

void test_allocate_and_free(void)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_EQUAL_INT(0, scnn_net_size(net));

    TEST_ASSERT_EQUAL_INT(1, scnn_net_batch_size(net));

    TEST_ASSERT_NULL(scnn_net_layers(net));

    TEST_ASSERT_NULL(scnn_net_input(net));
    TEST_ASSERT_NULL(scnn_net_output(net));

    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_free_pointer_to_NULL(void)
{
    scnn_net_free(&net);
}

void test_free_NULL(void)
{
    scnn_net_free(NULL);
}

void test_append_layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_ExpectAnyArgs();
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, param));

    TEST_ASSERT_EQUAL_INT(1, scnn_net_size(net));

    TEST_ASSERT_NOT_NULL(scnn_net_layers(net));

    TEST_ASSERT_EQUAL_INT(param.type, scnn_net_layers(net)[0].params.type);
    TEST_ASSERT_EQUAL_INT_ARRAY(param.in_shape, scnn_net_layers(net)[0].params.in_shape, 4);
    TEST_ASSERT_EQUAL_INT(param.out, scnn_net_layers(net)[0].params.out);

    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_output(net));

    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_3layers(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_ExpectAnyArgs();
    scnn_layer_connect_ExpectAnyArgs();
    scnn_layer_connect_ExpectAnyArgs();
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, params3[i]));
    }

    TEST_ASSERT_EQUAL_INT(3, scnn_net_size(net));

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_INT(params3[i].type, scnn_net_layers(net)[i].params.type);
        TEST_ASSERT_EQUAL_INT_ARRAY(params3[i].in_shape, scnn_net_layers(net)[i].params.in_shape, 4);
        TEST_ASSERT_EQUAL_INT(params3[i].out, scnn_net_layers(net)[i].params.out);
     }

    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[2], scnn_net_output(net));

    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_fail_if_net_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_net_append(NULL, param));
}

void test_append_fail_if_layer_is_NONE(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_ExpectAnyArgs();
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, param));

    scnn_layer *prev_layer = scnn_net_layers(net);

    TEST_ASSERT_NULL(scnn_net_append(net, (scnn_layer_params){}));

    TEST_ASSERT_EQUAL_INT(1, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(prev_layer, scnn_net_layers(net));

    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_output(net));

    scnn_net_free(&net);
}

void test_init(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, param);

    scnn_layer_init_ExpectAndReturn(&scnn_net_layers(net)[0], &scnn_net_layers(net)[0]);
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    scnn_net_free(&net);
}

void test_init_3layers(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, params3[i]);
    }

    for (int i = 0; i < 3; i++) {
        scnn_layer_init_ExpectAndReturn(&scnn_net_layers(net)[i], &scnn_net_layers(net)[i]);
    }
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    scnn_net_free(&net);
}

void test_init_fail_if_net_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_net_init(NULL));
}

void test_init_fail_if_net_size_is_0(void)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NULL(scnn_net_init(net));

    scnn_net_free(&net);
}

void test_init_fail_if_layer_init_fail(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, params3[i]);
    }

    scnn_layer_init_ExpectAndReturn(&scnn_net_layers(net)[0], &scnn_net_layers(net)[0]);
    scnn_layer_init_ExpectAndReturn(&scnn_net_layers(net)[1], NULL);
    TEST_ASSERT_NULL(scnn_net_init(net));

    scnn_net_free(&net);
}

void test_forward_1layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, param);

    scnn_layer_init_IgnoreAndReturn(&scnn_net_layers(net)[0]);
    scnn_net_init(net);

    scnn_dtype x[3 * 28 * 28];
    scnn_dtype y[100];
    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[0], x, y);
    TEST_ASSERT_EQUAL_PTR(y, scnn_net_forward(net, x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, params3[i]);
    }

    for (int i = 0; i < 3; i++) {
        scnn_layer_init_IgnoreAndReturn(&scnn_net_layers(net)[i]);
    }
    scnn_net_init(net);

    scnn_dtype x[3 * 28 * 28];
    scnn_dtype y0[100];
    scnn_dtype y1[10];
    scnn_dtype y2[10];
    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[0], x, y0);
    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[1], y0, y1);
    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[2], y1, y2);
    TEST_ASSERT_EQUAL_PTR(y2, scnn_net_forward(net, x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_fail_if_net_is_NULL(void)
{
    scnn_dtype x[3 * 28 * 28];
    TEST_ASSERT_NULL(scnn_net_forward(NULL, x));
}

void test_forward_fail_if_x_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, param);

    scnn_layer_init_IgnoreAndReturn(&scnn_net_layers(net)[0]);
    scnn_net_init(net);

    TEST_ASSERT_NULL(scnn_net_forward(net, NULL));

    scnn_net_free(&net);
}

void test_backward_1layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, param);

    scnn_dtype dx[3 * 28 * 28];
    scnn_dtype dy[100];
    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[0], dy, dx);
    TEST_ASSERT_EQUAL_PTR(dx, scnn_net_backward(net, dy));

    scnn_net_free(&net);
}

void test_backward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, params3[i]);
    }

    scnn_dtype dx0[3 * 28 * 28];
    scnn_dtype dx1[100];
    scnn_dtype dx2[10];
    scnn_dtype dy[10];
    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[2], dy, dx2);
    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[1], dx2, dx1);
    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[0], dx1, dx0);
    TEST_ASSERT_EQUAL_PTR(dx0, scnn_net_backward(net, dy));

    scnn_net_free(&net);
}

void test_backward_fail_if_net_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, param);

    scnn_dtype dy[100];
    TEST_ASSERT_NULL(scnn_net_backward(NULL, dy));

    scnn_net_free(&net);
}

void test_backward_fail_if_dy_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, param);

    TEST_ASSERT_NULL(scnn_net_backward(net, NULL));

    scnn_net_free(&net);
}
