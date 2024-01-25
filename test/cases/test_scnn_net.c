/**
 * @file test_scnn_net.c
 * @brief Unit tests of scnn_net.c
 * 
 */
#include "scnn_net.h"

#include "unity.h"

#include "mock_scnn_layer.h"
#include "scnn_layer_impl.h"

#define INIT_LAYER_SIZE 128

scnn_net *net;

scnn_layer dummy_layer;
scnn_layer dummy_layers[INIT_LAYER_SIZE];

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

    for (int i = 0; i < INIT_LAYER_SIZE; i++) {
        TEST_ASSERT_NULL(scnn_net_layers(net)[i]);
    }

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

    scnn_layer_connect_Expect(NULL, &dummy_layer);
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, &dummy_layer));

    TEST_ASSERT_EQUAL_INT(1, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(&dummy_layer, scnn_net_layers(net)[0]);

    TEST_ASSERT_EQUAL_PTR(&dummy_layer, scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&dummy_layer, scnn_net_output(net));

    scnn_layer_free_Expect(&(scnn_net_layers(net)[0]));
    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_3layers(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Expect(NULL, &dummy_layers[0]);
    scnn_layer_connect_Expect(&dummy_layers[0], &dummy_layers[1]);
    scnn_layer_connect_Expect(&dummy_layers[1], &dummy_layers[2]);
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, &dummy_layers[i]));
    }

    TEST_ASSERT_EQUAL_INT(3, scnn_net_size(net));

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_PTR(&dummy_layers[i], scnn_net_layers(net)[i]);
    }

    TEST_ASSERT_EQUAL_PTR(&dummy_layers[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&dummy_layers[2], scnn_net_output(net));

    for (int i = 0; i < 3; i++) {
        scnn_layer_free_Expect(&(scnn_net_layers(net)[i]));
    }
    scnn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_append_fail_if_net_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_net_append(NULL, &dummy_layer));
}

void test_append_fail_if_layer_is_NULL(void)
{
    net = scnn_net_alloc();

    TEST_ASSERT_NULL(scnn_net_append(net, NULL));

    TEST_ASSERT_EQUAL_INT(0, scnn_net_size(net));

    TEST_ASSERT_EQUAL_PTR(NULL, scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(NULL, scnn_net_output(net));

    scnn_net_free(&net);
}

void test_append_extend_layer_size(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Expect(NULL, &dummy_layers[0]);
    for (int i = 1; i < INIT_LAYER_SIZE; i++) {
        scnn_layer_connect_Expect(&dummy_layers[i - 1], &dummy_layers[i]);
    }
    for (int i = 0; i < INIT_LAYER_SIZE; i++) {
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, &dummy_layers[i]));
    }

    TEST_ASSERT_EQUAL_INT(INIT_LAYER_SIZE, scnn_net_size(net));
    TEST_ASSERT_EQUAL_PTR(&dummy_layers[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&dummy_layers[INIT_LAYER_SIZE - 1], scnn_net_output(net));

    scnn_layer extra_dummy_layer;
    scnn_layer_connect_Expect(&dummy_layers[INIT_LAYER_SIZE - 1], &extra_dummy_layer);
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, &extra_dummy_layer));

    TEST_ASSERT_EQUAL_INT((INIT_LAYER_SIZE + 1), scnn_net_size(net));
    TEST_ASSERT_EQUAL_PTR(&dummy_layers[0], scnn_net_input(net));
    TEST_ASSERT_EQUAL_PTR(&extra_dummy_layer, scnn_net_output(net));

    for (int i = 0; i < (INIT_LAYER_SIZE + 1); i++) {
        scnn_layer_free_Expect(&(scnn_net_layers(net)[i]));
    }
    scnn_net_free(&net);

    TEST_ASSERT_NULL(net);
}

void test_init(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, &dummy_layer);

    scnn_layer_init_ExpectAndReturn(&dummy_layer, &dummy_layer);
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_init_3layers(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &dummy_layers[i]);
    }

    for (int i = 0; i < 3; i++) {
        scnn_layer_init_ExpectAndReturn(&dummy_layers[i], &dummy_layers[i]);
    }
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_init(net));

    scnn_layer_free_Ignore();
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

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_init_fail_if_layer_init_fail(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &dummy_layers[i]);
    }

    scnn_layer_init_ExpectAndReturn(&dummy_layers[0], &dummy_layers[0]);
    scnn_layer_init_ExpectAndReturn(&dummy_layers[1], NULL);
    TEST_ASSERT_NULL(scnn_net_init(net));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_1layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, &dummy_layer);

    scnn_layer_init_IgnoreAndReturn(&dummy_layer);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_ExpectAndReturn(&dummy_layer, &x, &y);
    TEST_ASSERT_EQUAL_PTR(&y, scnn_net_forward(net, &x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &dummy_layers[i]);
    }

    for (int i = 0; i < 3; i++) {
        scnn_layer_init_IgnoreAndReturn(&dummy_layers[i]);
    }
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y[3];
    scnn_layer_forward_ExpectAndReturn(&dummy_layers[0], &x, &y[0]);
    scnn_layer_forward_ExpectAndReturn(&dummy_layers[1], &y[0], &y[1]);
    scnn_layer_forward_ExpectAndReturn(&dummy_layers[2], &y[1], &y[2]);

    TEST_ASSERT_EQUAL_PTR(&y[2], scnn_net_forward(net, &x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_fail_if_net_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, &dummy_layer);

    scnn_layer_init_IgnoreAndReturn(&dummy_layer);
    scnn_net_init(net);

    scnn_dtype x;
    TEST_ASSERT_NULL(scnn_net_forward(NULL, &x));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_forward_fail_if_x_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, &dummy_layer);

    scnn_layer_init_IgnoreAndReturn(&dummy_layer);
    scnn_net_init(net);

    scnn_dtype x;
    TEST_ASSERT_NULL(scnn_net_forward(net, NULL));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_1layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    scnn_net_append(net, &dummy_layer);

    scnn_layer_init_IgnoreAndReturn(&dummy_layer);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    scnn_dtype dy[2];
    scnn_layer_backward_ExpectAndReturn(&dummy_layer, &dy[1], &dy[0]);
    TEST_ASSERT_EQUAL_PTR(&dy[0], scnn_net_backward(net, &dy[1]));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_3layer(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &dummy_layers[i]);
    }

    scnn_layer_init_IgnoreAndReturn(&dummy_layers[0]);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    scnn_dtype dy[4];
    scnn_layer_backward_ExpectAndReturn(&dummy_layers[2], &dy[3], &dy[2]);
    scnn_layer_backward_ExpectAndReturn(&dummy_layers[1], &dy[2], &dy[1]);
    scnn_layer_backward_ExpectAndReturn(&dummy_layers[0], &dy[1], &dy[0]);
    TEST_ASSERT_EQUAL_PTR(&dy[0], scnn_net_backward(net, &dy[3]));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_fail_if_net_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &dummy_layers[i]);
    }

    scnn_layer_init_IgnoreAndReturn(&dummy_layers[0]);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    scnn_dtype dy[4];
    TEST_ASSERT_NULL(scnn_net_backward(NULL, &dy[3]));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}

void test_backward_fail_if_dy_is_NULL(void)
{
    net = scnn_net_alloc();

    scnn_layer_connect_Ignore();
    for (int i = 0; i < 3; i++) {
        scnn_net_append(net, &dummy_layers[i]);
    }

    scnn_layer_init_IgnoreAndReturn(&dummy_layers[0]);
    scnn_net_init(net);

    scnn_dtype x;
    scnn_dtype y;
    scnn_layer_forward_IgnoreAndReturn(&y);
    scnn_net_forward(net, &x);

    TEST_ASSERT_NULL(scnn_net_backward(net, NULL));

    scnn_layer_free_Ignore();
    scnn_net_free(&net);
}
