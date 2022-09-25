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

TEST(scnn_net, forward)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(&fc->w, (float[]){ 1, 2, 3, 4 }, fc->w.size);
    scnn_mat_copy_from_array(&fc->b, (float[]){ 0, 1 }, fc->w.size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_forward(net, x);

    float y_fc[] = { 0.7, 2 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_fc, net->layers[0]->y.data, net->layers[0]->y.size);

    float y_sigmoid[] = { 0.66818777, 0.88079708 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_sigmoid, net->layers[1]->y.data, net->layers[1]->y.size);

    float y_softmax[] = { 0.44704699, 0.55295301 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y_softmax, net->layers[2]->y.data, net->layers[2]->y.size);

    scnn_net_free(&net);
}

TEST(scnn_net, forward_net_is_null)
{
    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_forward(NULL, x);
}

TEST(scnn_net, forward_x_is_null)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(&fc->w, (float[]){ 1, 2, 3, 4 }, fc->w.size);
    scnn_mat_copy_from_array(&fc->b, (float[]){ 0, 1 }, fc->w.size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat_fill(&net->output->y, 0);

    scnn_net_forward(net, NULL);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, net->output->y.data, net->output->y.size);

    scnn_net_free(&net);
}

TEST(scnn_net, backward)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(&fc->w, (float[]){ 1, 2, 3, 4 }, fc->w.size);
    scnn_mat_copy_from_array(&fc->b, (float[]){ 0, 1 }, fc->w.size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_forward(net, x);

    scnn_mat *t = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(t, (float[]){ 0, 1 }, x->size);

    scnn_net_backward(net, t);

    float dx_ans[] = {
        0.00524194, 0.10959995
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx_ans, net->layers[0]->dx.data, net->layers[0]->y.size);

    scnn_net_free(&net);
}

TEST(scnn_net, backward_net_is_null)
{
    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 }});
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_net_backward(NULL, x);
}

TEST(scnn_net, backward_t_is_null)
{
    scnn_net *net = scnn_net_alloc();

    scnn_layer *fc = scnn_fc_layer((scnn_layer_params){ .in=2, .out=2 });
    fc->set_size(fc, 1, 2, 1, 1);
    scnn_layer *sigmoid = scnn_sigmoid_layer((scnn_layer_params){ .in=2 });
    sigmoid->set_size(sigmoid, 1, 2, 1, 1);
    scnn_layer *softmax = scnn_softmax_layer((scnn_layer_params){ .in=2 });
    softmax->set_size(softmax, 1, 2, 1, 1);

    scnn_mat_copy_from_array(&fc->w, (float[]){ 1, 2, 3, 4 }, fc->w.size);
    scnn_mat_copy_from_array(&fc->b, (float[]){ 0, 1 }, fc->w.size);

    scnn_net_append(net, fc);
    scnn_net_append(net, sigmoid);
    scnn_net_append(net, softmax);

    scnn_mat *x = scnn_mat_alloc((scnn_shape){ .d = { 1, 1, 1, 2 } });
    scnn_mat_copy_from_array(x, (float[]){ 0.1, 0.2 }, x->size);

    scnn_mat_fill(&net->input->dx, 0);

    scnn_net_forward(net, x);

    scnn_net_backward(net, NULL);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, net->input->dx.data, net->input->dx.size);

    scnn_net_free(&net);
}
