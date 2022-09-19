/**
 * @file test_scnn_net.c
 * @brief Unit tests of scnn_net.c
 * 
 */
#include "scnn_net.h"
#include "scnn_layers.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_net);

TEST_SETUP(scnn_net)
{}

TEST_TEAR_DOWN(scnn_net)
{}

TEST(scnn_net, alloc_and_free)
{
    scnn_net *net = scnn_net_alloc();

    TEST_ASSERT_NOT_NULL(net);

    TEST_ASSERT_EQUAL(0, net->size);

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        TEST_ASSERT_NULL(net->layers[i]);
    }

    TEST_ASSERT_NULL(net->input);
    TEST_ASSERT_NULL(net->output);

    scnn_net_free(&net);

    TEST_ASSERT_NULL(net);
}

TEST(scnn_net, append)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=10 });
    fc->set_size(fc, 1, 2, 1, 1);

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc));

    TEST_ASSERT_EQUAL(1, net->size);

    TEST_ASSERT_EQUAL_PTR(fc, net->layers[0]);

    TEST_ASSERT_EQUAL_PTR(fc, net->input);
    TEST_ASSERT_EQUAL_PTR(fc, net->output);

    scnn_net_free(&net);
}

TEST(scnn_net, append_2layers)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=10 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=10 });
    sigmoid->set_size(sigmoid, 1, 10, 1, 1);

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc));
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, sigmoid));

    TEST_ASSERT_EQUAL(2, net->size);

    TEST_ASSERT_EQUAL_PTR(fc, net->layers[0]);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->layers[1]);

    TEST_ASSERT_EQUAL_PTR(fc, net->input);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->output);

    scnn_net_free(&net);
}

TEST(scnn_net, append_3layers)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc1 = scnn_fc_layer((scnn_layer_params){ .in=2, .out=10 });
    fc1->set_size(fc1, 1, 2, 1, 1);
    scnn_layer *fc2 = scnn_fc_layer((scnn_layer_params){ .in=10, .out=10 });
    fc2->set_size(fc2, 1, 10, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=10 });
    sigmoid->set_size(sigmoid, 1, 10, 1, 1);

    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc1));
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc2));
    TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, sigmoid));

    TEST_ASSERT_EQUAL(3, net->size);

    TEST_ASSERT_EQUAL_PTR(fc1, net->layers[0]);
    TEST_ASSERT_EQUAL_PTR(fc2, net->layers[1]);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->layers[2]);

    TEST_ASSERT_EQUAL_PTR(fc1, net->input);
    TEST_ASSERT_EQUAL_PTR(sigmoid, net->output);

    scnn_net_free(&net);
}

TEST(scnn_net, append_net_is_null)
{
    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=10 });
    fc->set_size(fc, 1, 2, 1, 1);

    TEST_ASSERT_NULL(scnn_net_append(NULL, fc));
}

TEST(scnn_net, append_layer_is_null)
{
    scnn_net *net = scnn_net_alloc();

    TEST_ASSERT_NULL(scnn_net_append(net, NULL));

    scnn_net_free(&net);
}

TEST(scnn_net, append_unmatched_size)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=10 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=3 });
    sigmoid->set_size(sigmoid, 1, 3, 1, 1);

    scnn_net_append(net, fc);

    TEST_ASSERT_NULL(scnn_net_append(net, sigmoid));

    scnn_net_free(&net);
}

TEST(scnn_net, append_over_max_size)
{
    scnn_net *net = scnn_net_alloc();

    for (int i = 0; i < SCNN_NET_MAX_SIZE; i++) {
        scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
        fc->set_size(fc, 1, 2, 1, 1);
        TEST_ASSERT_EQUAL_PTR(net, scnn_net_append(net, fc));
    }

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    TEST_ASSERT_NULL(scnn_net_append(net, fc));

    scnn_net_free(&net);
}
