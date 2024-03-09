/**
 * @file test_nn_net.c
 * @brief Unit tests of nn_net.c
 *
 */
#include "nn_net.h"

#include <string.h>

#include "activation.h"
#include "blas.h"
#include "loss.h"
#include "nn_layer.h"

#include "unity.h"

NnLayerParams dummy_layer_params[] = {
    { .batch_size = 1, .in = 2, .out = 100 },
    { .out = 10 },
    { .out = 1 }
};

float dummy_x[3 * 28 * 28];
float dummy_dy[2];

void setUp(void) {}

void tearDown(void) {}

void test_allocate_and_free(void) {
    NnNet *net = nn_net_alloc();

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_EQUAL_INT(0, nn_net_size(net));

    TEST_ASSERT_NULL(nn_net_layers(net));

    TEST_ASSERT_NULL(nn_net_input(net));
    TEST_ASSERT_NULL(nn_net_output(net));

    nn_net_free(&net);
    TEST_ASSERT_NULL(net);
}

void test_free_pointer_to_NULL(void) {
    NnNet *net = NULL;
    nn_net_free(&net);
}

void test_free_NULL(void) {
    nn_net_free(NULL);
}

void test_allocate_and_free_layer(void) {
    NnNet net;
    TEST_ASSERT_EQUAL_PTR(
        &net,
        nn_net_alloc_layers(&net, 1,
            (NnLayerParams[]){ dummy_layer_params[0] }
        )
    );

    TEST_ASSERT_EQUAL_INT(1, nn_net_size(&net));

    TEST_ASSERT_NOT_NULL(nn_net_layers(&net));

    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].batch_size, nn_net_layers(&net)[0].batch_size);
    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].in, nn_net_layers(&net)[0].in);
    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].out, nn_net_layers(&net)[0].out);

    TEST_ASSERT_EQUAL_PTR(&nn_net_layers(&net)[0], nn_net_input(&net));
    TEST_ASSERT_EQUAL_PTR(&nn_net_layers(&net)[0], nn_net_output(&net));

    nn_net_free_layers(&net);
    TEST_ASSERT_NULL(net.layers);
}

void test_allocate_and_free_3layers(void) {
    NnNet net;
    TEST_ASSERT_EQUAL_PTR(
        &net,
        nn_net_alloc_layers(&net, 3, dummy_layer_params)
    );

    TEST_ASSERT_EQUAL_INT(3, nn_net_size(&net));

    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].batch_size, nn_net_layers(&net)[0].batch_size);
    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].in, nn_net_layers(&net)[0].in);
    TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].out, nn_net_layers(&net)[0].out);
    for (int i = 1; i < 3; i++) {
        TEST_ASSERT_EQUAL_INT(dummy_layer_params[0].batch_size, nn_net_layers(&net)[0].batch_size);
        TEST_ASSERT_EQUAL_INT(dummy_layer_params[i - 1].out, nn_net_layers(&net)[i].in);
        TEST_ASSERT_EQUAL_INT(dummy_layer_params[i].out, nn_net_layers(&net)[i].out);
    }

    TEST_ASSERT_EQUAL_PTR(&nn_net_layers(&net)[0], nn_net_input(&net));
    TEST_ASSERT_EQUAL_PTR(&nn_net_layers(&net)[2], nn_net_output(&net));

    nn_net_free_layers(&net);
    TEST_ASSERT_NULL(net.layers);
}

void test_allocation_fail_if_net_is_NULL(void) {
    TEST_ASSERT_NULL(nn_net_alloc_layers(NULL, 3, dummy_layer_params));
}

void test_allocation_fail_if_num_layer_is_not_positive(void) {
    NnNet net;
    TEST_ASSERT_NULL(nn_net_alloc_layers(&net, 0, dummy_layer_params));
    TEST_ASSERT_NULL(nn_net_alloc_layers(&net, -1, dummy_layer_params));
}

void test_allocation_fail_if_param_list_is_NULL(void) {
    NnNet net;
    TEST_ASSERT_NULL(nn_net_alloc_layers(&net, 1, NULL));
}

void test_free_layers_for_NULL(void) {
    NnNet *net = NULL;
    nn_net_free_layers(net);
}

void test_free_layers_when_layers_are_NULL(void) {
    NnNet net = { .layers = NULL };
    nn_net_free_layers(&net);
}

void test_init(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 1,
        (NnLayerParams[]){ dummy_layer_params[0] }
    );

    TEST_ASSERT_EQUAL_PTR(&net, nn_net_init(&net));

    nn_net_free_layers(&net);
}

void test_init_3layers(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 3, dummy_layer_params);

    TEST_ASSERT_EQUAL_PTR(&net, nn_net_init(&net));

    nn_net_free_layers(&net);
}

void test_init_fail_if_net_is_NULL(void) {
    TEST_ASSERT_NULL(nn_net_init(NULL));
}

void test_init_fail_if_net_size_is_0(void) {
    NnNet net = { .size = 0 };
    TEST_ASSERT_NULL(nn_net_init(&net));
}

static void fill_with_value(float *array, const float value, const size_t size) {
    for (int i = 0; i < size; i++) {
        array[i] = value;
    }
}

static void net_fill_with_value(NnNet *net, const float value) {
    for (int i = 0; i < net->size; i++) {
        fill_with_value(net->layers[i].w, value, (net->layers[i].in * net->layers[i].out));
        fill_with_value(net->layers[i].b, value, net->layers[i].out);
    }
}

void test_forward_1layer(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 1,
        (NnLayerParams[]){ dummy_layer_params[0] }
    );

    nn_net_init(&net);

    net_fill_with_value(&net, 0);

    TEST_ASSERT_EQUAL_PTR(nn_net_layers(&net)[0].z, nn_net_forward(&net, dummy_x));

    nn_net_free_layers(&net);
}

void test_forward_3layer(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 3, dummy_layer_params);

    nn_net_init(&net);

    net_fill_with_value(&net, 0);

    TEST_ASSERT_EQUAL_PTR(nn_net_layers(&net)[2].z, nn_net_forward(&net, dummy_x));

    nn_net_free_layers(&net);
}

void test_forward_fail_if_net_is_NULL(void) {
    TEST_ASSERT_NULL(nn_net_forward(NULL, dummy_x));
}

void test_forward_fail_if_x_is_NULL(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 1,
        (NnLayerParams[]){ dummy_layer_params[0] }
    );

    nn_net_init(&net);

    TEST_ASSERT_NULL(nn_net_forward(&net, NULL));

    nn_net_free_layers(&net);
}

void test_backward_1layer(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 1,
        (NnLayerParams[]){ dummy_layer_params[0] }
    );

    nn_net_init(&net);

    TEST_ASSERT_EQUAL_PTR(nn_net_layers(&net)[0].dx, nn_net_backward(&net, dummy_dy));

    nn_net_free_layers(&net);
}

void test_backward_3layer(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 3, dummy_layer_params);

    nn_net_init(&net);

    TEST_ASSERT_EQUAL_PTR(nn_net_layers(&net)[0].dx, nn_net_backward(&net, dummy_dy));

    nn_net_free_layers(&net);
}

void test_backward_fail_if_net_is_NULL(void) {
    TEST_ASSERT_NULL(nn_net_backward(NULL, dummy_dy));
}

void test_backward_fail_if_dy_is_NULL(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 1,
        (NnLayerParams[]){ dummy_layer_params[0] }
    );

    nn_net_init(&net);

    TEST_ASSERT_NULL(nn_net_backward(&net, NULL));

    nn_net_free_layers(&net);
}

static void verify_params_are_different(const float *a1, const float *a2, const size_t size) {
    for (int i = 0; i < size; i++) {
        TEST_ASSERT_TRUE(a1[i] != a2[i]);
    }
}

void test_update(void) {
    NnNet net;
    nn_net_alloc_layers(&net, 3, dummy_layer_params);

    nn_net_init(&net);

    net_fill_with_value(&net, 0.1);

    float x[] = { 0.1, 0.5 };
    nn_net_forward(&net, x);

    float dy[] = { 0.5 };
    nn_net_backward(&net, dy);

    // Store the weights/biases before update
    float prev_w1[100 * 2];
    float prev_b1[100];
    float prev_w2[10 * 100];
    float prev_b2[10];
    float prev_w3[1 * 10];
    float prev_b3[1];

    memcpy(prev_w1, net.layers[0].w, sizeof(prev_w1));
    memcpy(prev_b1, net.layers[0].b, sizeof(prev_b1));
    memcpy(prev_w2, net.layers[1].w, sizeof(prev_w2));
    memcpy(prev_b2, net.layers[1].b, sizeof(prev_b2));
    memcpy(prev_w3, net.layers[2].w, sizeof(prev_w3));
    memcpy(prev_b3, net.layers[2].b, sizeof(prev_b3));

    nn_net_update(&net, 0.1);

    verify_params_are_different(
        prev_w1, net.layers[0].w, (net.layers[0].in * net.layers[0].out)
    );
    verify_params_are_different(
        prev_b1, net.layers[0].b, net.layers[0].out
    );
    verify_params_are_different(
        prev_w2, net.layers[1].w, (net.layers[1].in * net.layers[1].out)
    );
    verify_params_are_different(
        prev_b2, net.layers[1].b, net.layers[1].out
    );
    verify_params_are_different(
        prev_w3, net.layers[2].w, (net.layers[2].in * net.layers[2].out)
    );
    verify_params_are_different(
        prev_b3, net.layers[2].b, net.layers[2].out
    );

    nn_net_free_layers(&net);
}
