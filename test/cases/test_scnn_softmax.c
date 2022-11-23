/**
 * @file test_scnn_softmax.c
 * @brief Unit tests of scnn_softmax.c
 *
 */
#include "scnn_softmax.h"

#include "scnn_blas.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_softmax);

scnn_layer *softmax;

TEST_SETUP(scnn_softmax)
{
    softmax = NULL;
}

TEST_TEAR_DOWN(scnn_softmax)
{
    scnn_layer_free(&softmax);

    TEST_ASSERT_NULL(softmax);
}

TEST(scnn_softmax, allocate_softmax_layer)
{
    scnn_layer_params params = {.in_shape = {1, 3, 1, 1}};
    softmax = scnn_softmax_layer(params);

    TEST_ASSERT_NOT_NULL(softmax);

    TEST_ASSERT_EQUAL_INT(1, softmax->params.in_shape[0]);
    TEST_ASSERT_EQUAL_INT(3, softmax->params.in_shape[1]);
    TEST_ASSERT_EQUAL_INT(1, softmax->params.in_shape[2]);
    TEST_ASSERT_EQUAL_INT(1, softmax->params.in_shape[3]);

    TEST_ASSERT_NOT_NULL(softmax->init);

    TEST_ASSERT_NOT_NULL(softmax->forward);
    TEST_ASSERT_NOT_NULL(softmax->backward);
}

TEST(scnn_softmax, initialize)
{
    scnn_layer_params params = {.in_shape = {1, 3, 1, 1}};
    softmax = scnn_softmax_layer(params);

    TEST_ASSERT_NOT_NULL(softmax->init(softmax));

    TEST_ASSERT_NOT_NULL(softmax->x);
    TEST_ASSERT_NOT_NULL(softmax->x->data);
    TEST_ASSERT_EQUAL_INT(1, softmax->x->shape[0]);
    TEST_ASSERT_EQUAL_INT(3, softmax->x->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, softmax->x->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, softmax->x->shape[3]);
    TEST_ASSERT_EQUAL_INT(3, softmax->x->size);

    TEST_ASSERT_NOT_NULL(softmax->y);
    TEST_ASSERT_NOT_NULL(softmax->y->data);
    TEST_ASSERT_EQUAL_INT(1, softmax->y->shape[0]);
    TEST_ASSERT_EQUAL_INT(3, softmax->y->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, softmax->y->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, softmax->y->shape[3]);
    TEST_ASSERT_EQUAL_INT(3, softmax->y->size);

    TEST_ASSERT_NULL(softmax->w);
    TEST_ASSERT_NULL(softmax->b);

    TEST_ASSERT_NOT_NULL(softmax->dx);
    TEST_ASSERT_NOT_NULL(softmax->dx->data);
    TEST_ASSERT_EQUAL_INT(softmax->x->shape[0], softmax->dx->shape[0]);
    TEST_ASSERT_EQUAL_INT(softmax->x->shape[1], softmax->dx->shape[1]);
    TEST_ASSERT_EQUAL_INT(softmax->x->shape[2], softmax->dx->shape[2]);
    TEST_ASSERT_EQUAL_INT(softmax->x->shape[3], softmax->dx->shape[3]);
    TEST_ASSERT_EQUAL_INT(softmax->x->size, softmax->dx->size);

    TEST_ASSERT_NULL(softmax->dw);
    TEST_ASSERT_NULL(softmax->db);
}

TEST(scnn_softmax, cannot_initialize_with_NULL)
{
    scnn_layer_params params = {.in_shape = {1, 3, 1, 1}};
    softmax = scnn_softmax_layer(params);

    TEST_ASSERT_NULL(softmax->init(NULL));

    TEST_ASSERT_NULL(softmax->x);
    TEST_ASSERT_NULL(softmax->y);
    TEST_ASSERT_NULL(softmax->w);
    TEST_ASSERT_NULL(softmax->b);
    TEST_ASSERT_NULL(softmax->dx);
    TEST_ASSERT_NULL(softmax->dw);
    TEST_ASSERT_NULL(softmax->db);
}

TEST(scnn_softmax, cannot_initialize_without_in_shape)
{
    scnn_layer_params params;
    softmax = scnn_softmax_layer(params);

    TEST_ASSERT_NULL(softmax->init(softmax));

    TEST_ASSERT_NULL(softmax->x);
    TEST_ASSERT_NULL(softmax->y);
    TEST_ASSERT_NULL(softmax->w);
    TEST_ASSERT_NULL(softmax->b);
    TEST_ASSERT_NULL(softmax->dx);
    TEST_ASSERT_NULL(softmax->dw);
    TEST_ASSERT_NULL(softmax->db);
}

TEST(scnn_softmax, cannot_initialize_with_invalid_in_shape)
{
    scnn_layer_params params = {.in_shape = {-1, 3, 1, 1}};
    softmax = scnn_softmax_layer(params);

    TEST_ASSERT_NULL(softmax->init(softmax));

    TEST_ASSERT_NULL(softmax->x);
    TEST_ASSERT_NULL(softmax->y);
    TEST_ASSERT_NULL(softmax->w);
    TEST_ASSERT_NULL(softmax->b);
    TEST_ASSERT_NULL(softmax->dx);
    TEST_ASSERT_NULL(softmax->dw);
    TEST_ASSERT_NULL(softmax->db);
}

TEST(scnn_softmax, forward)
{
    scnn_layer_params params = {.in_shape = {1, 3, 1, 1}};
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 1, 4};

    softmax->forward(softmax, x);

    scnn_dtype y[] = {
        0.00637746, 0.04712342, 0.94649912};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, softmax->y->data, 3);
}

TEST(scnn_softmax, forward_with_xy_dim)
{
    scnn_layer_params params = {.in_shape = {1, 3, 2, 2}};
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 0,
        -1, 1,
        1, 1,
        1, 1,
        4, 2,
        -2, 1,
    };

    softmax->forward(softmax, x);

    scnn_dtype y[] = {
        0.00637746, 0.09003057,
        0.1141952, 0.33333333,
        0.04712342, 0.24472847,
        0.84379473, 0.33333333,
        0.94649912, 0.66524096,
        0.04201007, 0.33333333
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, softmax->y->data, (3 * 2 * 2));
}

TEST(scnn_softmax, forward_with_batch_dim)
{
    scnn_layer_params params = { .in_shape = { 2, 3, 1, 1 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 1, 4,
        0, 1, 2
    };

    softmax->forward(softmax, x);

    scnn_dtype y[] = {
        0.00637746, 0.04712342, 0.94649912,
        0.09003057, 0.24472847, 0.66524096
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, softmax->y->data, (2 * 3));
}

TEST(scnn_softmax, forward_fails_when_layer_is_NULL)
{
    scnn_layer_params params = { .in_shape = { 1, 3, 1, 1 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype y[] = {
        0, 1, 2
    };
    scnn_scopy(3, y, 1, softmax->y->data, 1);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    softmax->forward(NULL, x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, softmax->y->data, 3);
}

TEST(scnn_softmax, forward_fails_when_x_is_NULL)
{
    scnn_layer_params params = { .in_shape = { 1, 3, 1, 1 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype y[] = {
        0, 1, 2
    };
    scnn_scopy(3, y, 1, softmax->y->data, 1);

    softmax->forward(softmax, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, softmax->y->data, 3);
}

TEST(scnn_softmax, backward)
{
    scnn_layer_params params = { .in_shape = { 1, 3, 1, 1 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 1, 4
    };

    softmax->forward(softmax, x);
    // scnn_dtype y[] = {
    //     0.00637746, 0.04712342, 0.94649912
    // };

    scnn_dtype t[] = {
        0, 0, 1
    };
    scnn_dtype dy[3];
    for (int i = 0; i < 3; i++) {
        dy[i] = softmax->y->data[i] - t[i];
    }

    softmax->backward(softmax, dy);

    scnn_dtype dx[] = {
        0.00637746, 0.04712342, -0.053500880
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, softmax->dx->data, 3);
}

TEST(scnn_softmax, backward_with_xy_dim)
{
    scnn_layer_params params = { .in_shape = { 1, 3, 2, 2 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 0,
        -1, 1,
        1, 1,
        1, 1,
        4, 2,
        -2, 1,
    };

    softmax->forward(softmax, x);

    // scnn_dtype y[] = {
    //     0.00637746, 0.09003057,
    //     0.1141952, 0.33333333,
    //     0.04712342, 0.24472847,
    //     0.84379473, 0.33333333,
    //     0.94649912, 0.66524096,
    //     0.04201007, 0.33333333
    // };

    scnn_dtype t[] = {
        0, 0,
        1, 0,
        0, 1,
        0, 0,
        1, 0,
        0, 1
    };
    scnn_dtype dy[3 * 2 * 2];
    for (int i = 0; i < (3 * 2 * 2); i++) {
        dy[i] = softmax->y->data[i] - t[i];
    }

    softmax->backward(softmax, dy);

    scnn_dtype dx[] = {
        0.00637746, 0.09003057,
        -0.8858048, 0.33333333,
        0.04712342, -0.75527153,
        0.84379473, 0.33333333,
        -0.05350088, 0.66524096,
        0.04201007, -0.66666667
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, softmax->dx->data, (3 * 2 * 2));
}

TEST(scnn_softmax, backward_with_batch_dim)
{
    scnn_layer_params params = { .in_shape = { 2, 3, 1, 1 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 1, 4,
        0, 1, 2
    };

    softmax->forward(softmax, x);
    // scnn_dtype y[] = {
    //     0.00637746, 0.04712342, 0.94649912,
    //     0.09003057, 0.24472847, 0.66524096
    // };

    scnn_dtype t[] = {
        0, 0, 1,
        0, 1, 0
    };
    scnn_dtype dy[2 * 3];
    for (int i = 0; i < (2 * 3); i++)
    {
        dy[i] = softmax->y->data[i] - t[i];
    }

    softmax->backward(softmax, dy);

    scnn_dtype dx[] = {
        0.00637746, 0.04712342, -0.05350087,
        0.09003057, -0.75527153, 0.66524096
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, softmax->dx->data, (2 * 3));
}

TEST(scnn_softmax, backward_fails_when_dy_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    softmax->forward(softmax, x);

    scnn_dtype dx[] = {
        0, 1, 2
    };

    scnn_scopy(3, dx, 1, softmax->dx->data, 1);

    softmax->backward(softmax, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, softmax->dx->data, softmax->dx->size);
}

TEST(scnn_softmax, backward_fails_when_layer_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 3, 1, 1 } };
    softmax = scnn_softmax_layer(params);
    softmax->init(softmax);

    scnn_dtype x[] = {
        -1, 0, 1
    };

    softmax->forward(softmax, x);

    scnn_dtype dx[] = {
        0, 1, 2
    };

    scnn_scopy(3, dx, 1, softmax->dx->data, 1);

    scnn_dtype dy[] = {
        0.53788284, 1, 1.46211716
    };

    softmax->backward(NULL, dy);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, softmax->dx->data, softmax->dx->size);
}
