/**
 * @file test_scnn_fc.c
 * @brief Unit tests of scnn_fc.c
 * 
 */
#include "scnn_fc.h"

#include "scnn_blas.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_fc);

TEST_SETUP(scnn_fc)
{}

TEST_TEAR_DOWN(scnn_fc)
{}

TEST(scnn_fc, alloc_and_free)
{
    scnn_layer_params params = { .in = 2, .out = 10 };
    scnn_layer *fc = scnn_fc_layer(params);

    TEST_ASSERT_NOT_NULL(fc);

    TEST_ASSERT_EQUAL_INT(params.in, fc->params.in);
    TEST_ASSERT_EQUAL_INT(params.out, fc->params.out);

    TEST_ASSERT_NOT_NULL(fc->forward);

    TEST_ASSERT_NOT_NULL(fc->set_size);

    scnn_layer_free(&fc);

    TEST_ASSERT_NULL(fc);
}

TEST(scnn_fc, alloc_fail_invalid_param_in)
{
    scnn_layer_params params = { .in = 0, .out = 10 };
    scnn_layer *fc = scnn_fc_layer(params);

    TEST_ASSERT_NULL(fc);
}

TEST(scnn_fc, alloc_fail_invalid_param_out)
{
    scnn_layer_params params = { .in = 2, .out = 0 };
    scnn_layer *fc = scnn_fc_layer(params);

    TEST_ASSERT_NULL(fc);
}
