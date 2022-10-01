/**
 * @file test_scnn_sigmoid.c
 * @brief Unit tests of scnn_sigmoid.c
 * 
 */
#include "scnn_sigmoid.h"

#include "scnn_blas.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_sigmoid);

TEST_SETUP(scnn_sigmoid)
{}

TEST_TEAR_DOWN(scnn_sigmoid)
{}

TEST(scnn_sigmoid, alloc_and_free)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    TEST_ASSERT_NOT_NULL(sigmoid);

    TEST_ASSERT_EQUAL_INT(params.in, sigmoid->params.in);
    TEST_ASSERT_EQUAL_INT(params.in, sigmoid->params.out);
    TEST_ASSERT_EQUAL_INT(sigmoid->params.in, sigmoid->params.out);

    TEST_ASSERT_NOT_NULL(sigmoid->forward);
    TEST_ASSERT_NOT_NULL(sigmoid->backward);

    TEST_ASSERT_NOT_NULL(sigmoid->set_size);

    scnn_layer_free(&sigmoid);

    TEST_ASSERT_NULL(sigmoid);
}

TEST(scnn_sigmoid, alloc_fail_invalid_param_in)
{
    scnn_layer_params params = { .in = 0 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    TEST_ASSERT_NULL(sigmoid);
}

TEST(scnn_sigmoid, set_size)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    sigmoid->set_size(sigmoid, 1, 10, 1, 1);

    TEST_ASSERT_NOT_NULL(sigmoid->x.data);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->x.shape.d[0]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->x.shape.d[1]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->x.shape.d[2]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->x.shape.d[3]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->x.size);

    TEST_ASSERT_NOT_NULL(sigmoid->y.data);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->y.shape.d[0]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->y.shape.d[1]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->y.shape.d[2]);
    TEST_ASSERT_EQUAL_INT(1, sigmoid->y.shape.d[3]);
    TEST_ASSERT_EQUAL_INT(10, sigmoid->y.size);

    TEST_ASSERT_NULL(sigmoid->w.data);

    TEST_ASSERT_NULL(sigmoid->b.data);

    TEST_ASSERT_NOT_NULL(sigmoid->dx.data);
    TEST_ASSERT_EQUAL_INT(sigmoid->x.shape.d[0], sigmoid->dx.shape.d[0]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x.shape.d[1], sigmoid->dx.shape.d[1]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x.shape.d[2], sigmoid->dx.shape.d[2]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x.shape.d[3], sigmoid->dx.shape.d[3]);
    TEST_ASSERT_EQUAL_INT(sigmoid->x.size, sigmoid->dx.size);

    TEST_ASSERT_NULL(sigmoid->dw.data);

    TEST_ASSERT_NULL(sigmoid->db.data);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, set_size_fail_invalid_n)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    sigmoid->set_size(sigmoid, 0, 10, 1, 1);

    TEST_ASSERT_NULL(sigmoid->x.data);
    TEST_ASSERT_NULL(sigmoid->y.data);
    TEST_ASSERT_NULL(sigmoid->dx.data);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, set_size_fail_invalid_c)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    sigmoid->set_size(sigmoid, 1, 0, 1, 1);

    TEST_ASSERT_NULL(sigmoid->x.data);
    TEST_ASSERT_NULL(sigmoid->y.data);
    TEST_ASSERT_NULL(sigmoid->dx.data);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, set_size_fail_invalid_h)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    sigmoid->set_size(sigmoid, 1, 10, 0, 1);

    TEST_ASSERT_NULL(sigmoid->x.data);
    TEST_ASSERT_NULL(sigmoid->y.data);
    TEST_ASSERT_NULL(sigmoid->dx.data);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, set_size_fail_invalid_w)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    sigmoid->set_size(sigmoid, 1, 10, 1, 0);

    TEST_ASSERT_NULL(sigmoid->x.data);
    TEST_ASSERT_NULL(sigmoid->y.data);
    TEST_ASSERT_NULL(sigmoid->dx.data);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, set_size_fail_invalid_in_size)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);

    sigmoid->set_size(sigmoid, 1, 10, 3, 3);

    TEST_ASSERT_NULL(sigmoid->x.data);
    TEST_ASSERT_NULL(sigmoid->y.data);
    TEST_ASSERT_NULL(sigmoid->dx.data);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, forward)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);
    sigmoid->set_size(sigmoid, 1, 3, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 1
        },
        sigmoid->x.size);

    sigmoid->forward(sigmoid, &x);

    float answer[] = {
        0.268941, 0.5, 0.731059
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, sigmoid->y.data, 3);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, forward_fail_x_is_null)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);
    sigmoid->set_size(sigmoid, 1, 3, 1, 1);

    float init[] = {
        0, 0, 0
    };

    scnn_mat_copy_from_array(&sigmoid->y,
        init,
        sigmoid->y.size);

    sigmoid->forward(sigmoid, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(init, sigmoid->y.data, sigmoid->y.size);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, forward_fail_layer_is_null)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);
    sigmoid->set_size(sigmoid, 1, 3, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 2, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 1
        },
        sigmoid->x.size);

    float init[] = {
        0, 0, 0
    };

    scnn_mat_copy_from_array(&sigmoid->y,
        init,
        sigmoid->y.size);

    sigmoid->forward(NULL, &x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(init, sigmoid->y.data, sigmoid->y.size);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, backward)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);
    sigmoid->set_size(sigmoid, 1, 3, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 1
        },
        sigmoid->x.size);

    sigmoid->forward(sigmoid, &x);

    scnn_mat dy;
    scnn_mat_init(&dy, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&dy,
        (float[]){
            0.53788284, 1, 1.46211716
        },
        dy.size);

    sigmoid->backward(sigmoid, &dy);

    float answer_dx[] = {
        1.05754186e-1, 2.5e-1, 2.87469681e-1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer_dx, sigmoid->dx.data, sigmoid->dx.size);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, backward_fail_dy_is_null)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);
    sigmoid->set_size(sigmoid, 1, 3, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 1
        },
        sigmoid->x.size);

    scnn_mat_fill(&sigmoid->dx, 0);

    sigmoid->backward(sigmoid, NULL);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, sigmoid->dx.data, sigmoid->dx.size);

    scnn_layer_free(&sigmoid);
}

TEST(scnn_sigmoid, backward_fail_layer_is_null)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *sigmoid = scnn_sigmoid_layer(params);
    sigmoid->set_size(sigmoid, 1, 3, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 1
        },
        sigmoid->x.size);

    scnn_mat dy;
    scnn_mat_init(&dy, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&dy,
        (float[]){
            8, 12, 16
        },
        dy.size);

    scnn_mat_fill(&sigmoid->dx, 0);

    sigmoid->backward(NULL, &dy);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, sigmoid->dx.data, sigmoid->dx.size);

    scnn_layer_free(&sigmoid);
}
