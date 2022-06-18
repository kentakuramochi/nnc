/**
 * @file test_scnn_mat.c
 * @brief Unit tests of scnn_mat.c
 * 
 */
#include "scnn_mat.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_mat);

TEST_SETUP(scnn_mat)
{}

TEST_TEAR_DOWN(scnn_mat)
{}

TEST(scnn_mat, alloc_and_free)
{
    scnn_mat *mat = scnn_mat_alloc(1, 3, 480, 640);

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->n);
    TEST_ASSERT_EQUAL(3, mat->c);
    TEST_ASSERT_EQUAL(480, mat->h);
    TEST_ASSERT_EQUAL(640, mat->w);

    TEST_ASSERT_EQUAL((1 * 3 * 480 * 640), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);

    scnn_mat_free(&mat);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_n_zero)
{
    scnn_mat *mat = scnn_mat_alloc(0, 3, 480, 640);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_c_zero)
{
    scnn_mat *mat = scnn_mat_alloc(1, 0, 480, 640);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_h_zero)
{
    scnn_mat *mat = scnn_mat_alloc(1, 3, 0, 640);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_w_zero)
{
    scnn_mat *mat = scnn_mat_alloc(1, 3, 480, 0);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_n_negative)
{
    scnn_mat *mat = scnn_mat_alloc(-1, 3, 480, 640);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_c_negative)
{
    scnn_mat *mat = scnn_mat_alloc(1, -1, 480, 640);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_h_negative)
{
    scnn_mat *mat = scnn_mat_alloc(1, 3, -1, 640);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_fail_w_negative)
{
    scnn_mat *mat = scnn_mat_alloc(1, 3, 480, -1);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, free_to_null)
{
    scnn_mat_free(NULL);
}

TEST(scnn_mat, free_to_ptr_to_null)
{
    scnn_mat *mat = NULL;

    scnn_mat_free(&mat);
}
