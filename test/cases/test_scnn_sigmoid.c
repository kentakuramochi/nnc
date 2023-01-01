/**
 * @file test_scnn_sigmoid.c
 * @brief Unit tests of scnn_sigmoid.c
 * 
 */
#include "scnn_sigmoid.h"

#include "scnn_blas.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_sigmoid);

scnn_layer *sigmoid;

TEST_SETUP(scnn_sigmoid)
{
    sigmoid = NULL;
}

TEST_TEAR_DOWN(scnn_sigmoid)
{
    scnn_layer_free(&sigmoid);

    TEST_ASSERT_NULL(sigmoid);
}

TEST(scnn_sigmoid, allocate_sigmoid_layer)
{
    scnn_layer_params params = { .in_shape={ 1, 10, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);

    TEST_ASSERT_NOT_NULL(sigmoid);

    TEST_ASSERT_EQUAL(SCNN_LAYER_SIGMOID, sigmoid->params.type);

    TEST_ASSERT_EQUAL_INT(1, sigmoid->params.in_shape[0]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->params.in_shape[1]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->params.in_shape[2]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->params.in_shape[3]);

    TEST_ASSERT_NOT_NULL(sigmoid->init);

    TEST_ASSERT_NOT_NULL(sigmoid->forward);
    TEST_ASSERT_NOT_NULL(sigmoid->backward);
}

TEST(scnn_sigmoid, initialize)
{
    scnn_layer_params params = { .in_shape={ 1, 10, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);

    TEST_ASSERT_NOT_NULL(sigmoid->init(sigmoid));

    TEST_ASSERT_NOT_NULL(sigmoid->x);
    TEST_ASSERT_NOT_NULL(sigmoid->x->data);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->x->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->x->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->x->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->x->shape[3]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->x->size);

    TEST_ASSERT_NOT_NULL(sigmoid->y);
    TEST_ASSERT_NOT_NULL(sigmoid->y->data);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->y->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->y->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->y->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->y->shape[3]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->y->size);

    TEST_ASSERT_NULL(sigmoid->w);
    TEST_ASSERT_NULL(sigmoid->b);

    TEST_ASSERT_NOT_NULL(sigmoid->dx);
    TEST_ASSERT_NOT_NULL(sigmoid->dx->data);
    TEST_ASSERT_EQUAL_INT(sigmoid->x->shape[0], sigmoid->dx->shape[0]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x->shape[1], sigmoid->dx->shape[1]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x->shape[2], sigmoid->dx->shape[2]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x->shape[3], sigmoid->dx->shape[3]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x->size, sigmoid->dx->size);

    TEST_ASSERT_NULL(sigmoid->dw);
    TEST_ASSERT_NULL(sigmoid->db);
}

TEST(scnn_sigmoid, cannot_initialize_with_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 10, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);

    TEST_ASSERT_NULL(sigmoid->init(NULL));

    TEST_ASSERT_NULL(sigmoid->x);
    TEST_ASSERT_NULL(sigmoid->y);
    TEST_ASSERT_NULL(sigmoid->w);
    TEST_ASSERT_NULL(sigmoid->b);
    TEST_ASSERT_NULL(sigmoid->dx);
    TEST_ASSERT_NULL(sigmoid->dw);
    TEST_ASSERT_NULL(sigmoid->db);
}

TEST(scnn_sigmoid, cannot_initialize_with_invalid_in_shape)
{
    scnn_layer_params params = { .in_shape={ -1, 10, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);

    TEST_ASSERT_NULL(sigmoid->init(sigmoid));

    TEST_ASSERT_NULL(sigmoid->x);
    TEST_ASSERT_NULL(sigmoid->y);
    TEST_ASSERT_NULL(sigmoid->w);
    TEST_ASSERT_NULL(sigmoid->b);
    TEST_ASSERT_NULL(sigmoid->dx);
    TEST_ASSERT_NULL(sigmoid->dw);
    TEST_ASSERT_NULL(sigmoid->db);
}

TEST(scnn_sigmoid, forward)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    sigmoid->forward(sigmoid, x);

    scnn_dtype y[] = {
        0.268941, 0.5, 0.731059
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, sigmoid->y->data, 3);
}

TEST(scnn_sigmoid, forward_with_batch_dim)
{
    scnn_layer_params params = { .in_shape={ 2, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype x[] = {
        -2, -1, 0,
        1, 2, 3
    };

    sigmoid->forward(sigmoid, x);

    scnn_dtype y[] = {
        0.119203, 0.268941, 0.5,
        0.731059, 0.880797, 0.952574
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, sigmoid->y->data, (2 * 3));
}

TEST(scnn_sigmoid, forward_fails_when_x_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype y[] = {
        0, 1, 2
    };
    scnn_scopy(3, y, 1, sigmoid->y->data, 1);

    sigmoid->forward(sigmoid, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, sigmoid->y->data, 3);
}

TEST(scnn_sigmoid, forward_fails_when_layer_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype y[] = {
        0, 1, 2
    };
    scnn_scopy(3, y, 1, sigmoid->y->data, 1);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    sigmoid->forward(NULL, x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, sigmoid->y->data, 3);
}

TEST(scnn_sigmoid, backward)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    sigmoid->forward(sigmoid, x);

    scnn_dtype dy[] = {
        0.53788284, 1, 1.46211716
    };

    sigmoid->backward(sigmoid, dy);

    scnn_dtype dx[] = {
        1.05754186e-1, 2.5e-1, 2.87469681e-1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, sigmoid->dx->data, sigmoid->dx->size);
}

TEST(scnn_sigmoid, backward_with_batch_dim)
{
    scnn_layer_params params = { .in_shape={ 2, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype x[] = {
        -2, -1, 0,
        1, 2, 3
    };

    sigmoid->forward(sigmoid, x);

    scnn_dtype dy[] = {
        0.238406, 0.53788284, 1,
        1.46211716, 1.761594, 1.905148
    };

    sigmoid->backward(sigmoid, dy);

    scnn_dtype dx[] = {
        2.503111e-2, 1.05754186e-1, 2.5e-1,
        2.87469681e-1, 1.8495617e-1, 8.606844e-2
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, sigmoid->dx->data, (2 * 3));
}

TEST(scnn_sigmoid, backward_fails_when_dy_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    sigmoid->forward(sigmoid, x);

    scnn_dtype dx[] = {
        0, 1, 2
    };

    scnn_scopy(3, dx, 1, sigmoid->dx->data, 1);

    sigmoid->backward(sigmoid, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, sigmoid->dx->data, sigmoid->dx->size);
}

TEST(scnn_sigmoid, backward_fails_when_layer_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    sigmoid = scnn_sigmoid_layer(params);
    sigmoid->init(sigmoid);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    sigmoid->forward(sigmoid, x);

    scnn_dtype dx[] = {
        0, 1, 2
    };

    scnn_scopy(3, dx, 1, sigmoid->dx->data, 1);

    scnn_dtype dy[] = {
        0.53788284, 1, 1.46211716
    };

    sigmoid->backward(NULL, dy);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, sigmoid->dx->data, sigmoid->dx->size);
}
