/**
 * @file test_scnn_softmax.c
 * @brief Unit tests of scnn_softmax.c
 * 
 */
#include "scnn_softmax.h"

#include "scnn_blas.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_softmax);

TEST_SETUP(scnn_softmax)
{}

TEST_TEAR_DOWN(scnn_softmax)
{}

TEST(scnn_softmax, alloc_and_free)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    TEST_ASSERT_NOT_NULL(softmax);

    TEST_ASSERT_EQUAL_INT(params.in, softmax->params.in);
    TEST_ASSERT_EQUAL_INT(params.in, softmax->params.out);
    TEST_ASSERT_EQUAL_INT(softmax->params.in, softmax->params.out);

    TEST_ASSERT_NOT_NULL(softmax->forward);
    TEST_ASSERT_NOT_NULL(softmax->backward);

    TEST_ASSERT_NOT_NULL(softmax->set_size);

    scnn_layer_free(&softmax);

    TEST_ASSERT_NULL(softmax);
}

TEST(scnn_softmax, alloc_fail_invalid_param_in)
{
    scnn_layer_params params = { .in = 0 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    TEST_ASSERT_NULL(softmax);
}

TEST(scnn_softmax, set_size)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    softmax->set_size(softmax, 1, 10, 1, 1);

    TEST_ASSERT_NOT_NULL(softmax->x.data);
    TEST_ASSERT_EQUAL_INT(1, softmax->x.n);
    TEST_ASSERT_EQUAL_INT(10, softmax->x.c);
    TEST_ASSERT_EQUAL_INT(1, softmax->x.h);
    TEST_ASSERT_EQUAL_INT(1, softmax->x.w);
    TEST_ASSERT_EQUAL_INT(10, softmax->x.size);

    TEST_ASSERT_NOT_NULL(softmax->y.data);
    TEST_ASSERT_EQUAL_INT(1, softmax->y.n);
    TEST_ASSERT_EQUAL_INT(10, softmax->y.c);
    TEST_ASSERT_EQUAL_INT(1, softmax->y.h);
    TEST_ASSERT_EQUAL_INT(1, softmax->y.w);
    TEST_ASSERT_EQUAL_INT(10, softmax->y.size);

    TEST_ASSERT_NULL(softmax->w.data);

    TEST_ASSERT_NULL(softmax->b.data);

    TEST_ASSERT_NOT_NULL(softmax->dx.data);
    TEST_ASSERT_EQUAL_INT(softmax->x.n, softmax->dx.n);
    TEST_ASSERT_EQUAL_INT(softmax->x.c, softmax->dx.c);
    TEST_ASSERT_EQUAL_INT(softmax->x.h, softmax->dx.h);
    TEST_ASSERT_EQUAL_INT(softmax->x.w, softmax->dx.w);
    TEST_ASSERT_EQUAL_INT(softmax->x.size, softmax->dx.size);

    TEST_ASSERT_NULL(softmax->dw.data);

    TEST_ASSERT_NULL(softmax->db.data);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, set_size_fail_invalid_n)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    softmax->set_size(softmax, 0, 10, 1, 1);

    TEST_ASSERT_NULL(softmax->x.data);
    TEST_ASSERT_NULL(softmax->y.data);
    TEST_ASSERT_NULL(softmax->dx.data);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, set_size_fail_invalid_c)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    softmax->set_size(softmax, 1, 0, 1, 1);

    TEST_ASSERT_NULL(softmax->x.data);
    TEST_ASSERT_NULL(softmax->y.data);
    TEST_ASSERT_NULL(softmax->dx.data);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, set_size_fail_invalid_h)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    softmax->set_size(softmax, 1, 10, 0, 1);

    TEST_ASSERT_NULL(softmax->x.data);
    TEST_ASSERT_NULL(softmax->y.data);
    TEST_ASSERT_NULL(softmax->dx.data);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, set_size_fail_invalid_w)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    softmax->set_size(softmax, 1, 10, 1, 0);

    TEST_ASSERT_NULL(softmax->x.data);
    TEST_ASSERT_NULL(softmax->y.data);
    TEST_ASSERT_NULL(softmax->dx.data);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, set_size_fail_invalid_in_size)
{
    scnn_layer_params params = { .in = 10 };
    scnn_layer *softmax = scnn_softmax_layer(params);

    softmax->set_size(softmax, 1, 10, 3, 3);

    TEST_ASSERT_NULL(softmax->x.data);
    TEST_ASSERT_NULL(softmax->y.data);
    TEST_ASSERT_NULL(softmax->dx.data);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, forward)
{
    scnn_layer_params params = { .in = 4 };
    scnn_layer *softmax = scnn_softmax_layer(params);
    softmax->set_size(softmax, 1, 4, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 4, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 3, 5
        },
        softmax->x.size);

    softmax->forward(softmax, &x);

    float answer[] = {
        0.0021657, 0.00588697, 0.11824302, 0.87370431
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, softmax->y.data, 4);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, forward_fail_x_is_null)
{
    scnn_layer_params params = { .in = 4 };
    scnn_layer *softmax = scnn_softmax_layer(params);
    softmax->set_size(softmax, 1, 4, 1, 1);

    float init[] = {
        0, 0, 0, 0
    };

    scnn_mat_copy_from_array(&softmax->y,
        init,
        softmax->y.size);

    softmax->forward(softmax, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(init, softmax->y.data, softmax->y.size);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, forward_fail_layer_is_null)
{
    scnn_layer_params params = { .in = 4 };
    scnn_layer *softmax = scnn_softmax_layer(params);
    softmax->set_size(softmax, 1, 4, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 4, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 3, 5
        },
        softmax->x.size);

    float init[] = {
        0, 0, 0, 0
    };

    scnn_mat_copy_from_array(&softmax->y,
        init,
        softmax->y.size);

    softmax->forward(NULL, &x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(init, softmax->y.data, softmax->y.size);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, backward)
{
    scnn_layer_params params = { .in = 4 };
    scnn_layer *softmax = scnn_softmax_layer(params);
    softmax->set_size(softmax, 1, 4, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 4, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 3, 5
        },
        softmax->x.size);

    softmax->forward(softmax, &x);

    float t[] = {
        0, 0, 1, 0
    };

    scnn_mat dy;
    scnn_mat_init(&dy, 1, 4, 1, 1);
    for (int i = 0; i < dy.size; i++)
    {
        dy.data[i] = softmax->y.data[i] - t[i];
    }

    softmax->backward(softmax, &dy);

    float answer_dx[] = {
        0.0021657, 0.00588697, -0.88175698, 0.87370431
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer_dx, softmax->dx.data, softmax->dx.size);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, backward_fail_dy_is_null)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *softmax = scnn_softmax_layer(params);
    softmax->set_size(softmax, 1, 3, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 1
        },
        softmax->x.size);

    scnn_mat_fill(&softmax->dx, 0);

    softmax->backward(softmax, NULL);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, softmax->dx.data, softmax->dx.size);

    scnn_layer_free(&softmax);
}

TEST(scnn_softmax, backward_fail_layer_is_null)
{
    scnn_layer_params params = { .in = 3 };
    scnn_layer *softmax = scnn_softmax_layer(params);
    softmax->set_size(softmax, 1, 3, 1, 1);

    scnn_mat x;
    scnn_mat_init(&x, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&x,
        (float[]){
            -1, 0, 1
        },
        softmax->x.size);

    scnn_mat dy;
    scnn_mat_init(&dy, 1, 3, 1, 1);
    scnn_mat_copy_from_array(&dy,
        (float[]){
            8, 12, 16
        },
        dy.size);

    scnn_mat_fill(&softmax->dx, 0);

    softmax->backward(NULL, &dy);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, softmax->dx.data, softmax->dx.size);

    scnn_layer_free(&softmax);
}
