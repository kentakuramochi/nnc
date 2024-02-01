/**
 * @file test_scnn_net.c
 * @brief Unit tests of scnn_net.c
 * 
 */
#include "scnn_net.h"

#include "unity.h"

#include "mock_scnn_layer.h"

scnn_net *net;

scnn_layer_params dummy_layer_params[] = {
    { .in = 3 * 28 * 28, .out = 100 },
    { .out = 10 },
    { .out = 2 }
};

scnn_dtype dummy_x[3 * 28 * 28];
scnn_dtype dummy_y0[100];
scnn_dtype dummy_y1[10];
scnn_dtype dummy_y2[2];

scnn_dtype dummy_dx0[3 * 28 * 28];
scnn_dtype dummy_dx1[100];
scnn_dtype dummy_dx2[10];
scnn_dtype dummy_dy[2];

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
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, dummy_layer_params[0]));

    TEST_ASSERT_EQUAL_INT(1, scnn_net_size(net));

    TEST_ASSERT_NOT_NULL(scnn_net_layers(net));

    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].in, scnn_net_layers(net)[0].params.in);
    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].out, scnn_net_layers(net)[0].params.out);

    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_output(net));

    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_3layers(void)
{
    net = scnn_net_alloc();

    for (int i = 0; i < 3; i++) {
        scnn_layer_connect_ExpectAnyArgs();
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, dummy_layer_params[i]));
    }

    TEST_ASSERT_EQUAL_INT(3, scnn_net_size(net));

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_INT(dummy_layer_params[i].in, scnn_net_layers(net)[i].params.in);
        TEST_ASSERT_EQUAL_INT(dummy_layer_params[i].out, scnn_net_layers(net)[i].params.out);
     }

    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&scnn_net_layers(net)[2], scnn_net_output(net));

    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_fail_if_net_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_net_append(NULL, dummy_layer_params[0]));
}

void test_init(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, dummy_layer_params[0]);

    scnn_layer_init_ExpectAndReturn(&scnn_net_layers(net)[0], &scnn_net_layers(net)[0]);
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    scnn_net_free(&net);
}

void test_init_3layers(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, dummy_layer_params[i]);
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
        scnn_net_append(net, dummy_layer_params[i]);
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
    scnn_net_append(net, dummy_layer_params[0]);

    scnn_layer_init_IgnoreAndReturn(&scnn_net_layers(net)[0]);
    scnn_net_init(net);

    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[0], dummy_x, dummy_y0);
    TEST_ASSERT_EQUAL_PTR(dummy_y0, scnn_net_forward(net, dummy_x));

    scnn_net_free(&net);
}

void test_forward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, dummy_layer_params[i]);
    }

    for (int i = 0; i < 3; i++) {
        scnn_layer_init_IgnoreAndReturn(&scnn_net_layers(net)[i]);
    }
    scnn_net_init(net);

    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[0], dummy_x, dummy_y0);
    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[1], dummy_y0, dummy_y1);
    scnn_layer_forward_ExpectAndReturn(&scnn_net_layers(net)[2], dummy_y1, dummy_y2);
    TEST_ASSERT_EQUAL_PTR(dummy_y2, scnn_net_forward(net, dummy_x));

    scnn_net_free(&net);
}

void test_forward_fail_if_net_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_net_forward(NULL, dummy_x));
}

void test_forward_fail_if_x_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, dummy_layer_params[0]);

    scnn_layer_init_IgnoreAndReturn(&scnn_net_layers(net)[0]);
    scnn_net_init(net);

    TEST_ASSERT_NULL(scnn_net_forward(net, NULL));

    scnn_net_free(&net);
}

void test_backward_1layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, dummy_layer_params[0]);

    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[0], dummy_dy, dummy_dx0);
    TEST_ASSERT_EQUAL_PTR(dummy_dx0, scnn_net_backward(net, dummy_dy));

    scnn_net_free(&net);
}

void test_backward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, dummy_layer_params[i]);
    }

    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[2], dummy_dy, dummy_dx2);
    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[1], dummy_dx2, dummy_dx1);
    scnn_layer_backward_ExpectAndReturn(&scnn_net_layers(net)[0], dummy_dx1, dummy_dx0);
    TEST_ASSERT_EQUAL_PTR(dummy_dx0, scnn_net_backward(net, dummy_dy));

    scnn_net_free(&net);
}

void test_backward_fail_if_net_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, dummy_layer_params[0]);

    TEST_ASSERT_NULL(scnn_net_backward(NULL, dummy_dy));

    scnn_net_free(&net);
}

void test_backward_fail_if_dy_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, dummy_layer_params[0]);

    TEST_ASSERT_NULL(scnn_net_backward(net, NULL));

    scnn_net_free(&net);
}
