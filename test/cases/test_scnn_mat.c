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
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 1, 3, 28, 28 } });

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->shape.d[0]);
    TEST_ASSERT_EQUAL(3, mat->shape.d[1]);
    TEST_ASSERT_EQUAL(28, mat->shape.d[2]);
    TEST_ASSERT_EQUAL(28, mat->shape.d[3]);

    TEST_ASSERT_EQUAL((1 * 3 * 28 * 28), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);

    scnn_mat_free(&mat);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, alloc_shape_1d)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 28 } });

    TEST_ASSERT_EQUAL(1, mat->shape.d[0]);
    TEST_ASSERT_EQUAL(1, mat->shape.d[1]);
    TEST_ASSERT_EQUAL(1, mat->shape.d[2]);
    TEST_ASSERT_EQUAL(28, mat->shape.d[3]);

    TEST_ASSERT_EQUAL(28, mat->size);

    TEST_ASSERT_NOT_NULL(mat->data);

    scnn_mat_free(&mat);
}

TEST(scnn_mat, alloc_shape_2d)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 28, 28 } });

    TEST_ASSERT_EQUAL(1, mat->shape.d[0]);
    TEST_ASSERT_EQUAL(1, mat->shape.d[1]);
    TEST_ASSERT_EQUAL(28, mat->shape.d[2]);
    TEST_ASSERT_EQUAL(28, mat->shape.d[3]);

    TEST_ASSERT_EQUAL((28 * 28), mat->size);

    TEST_ASSERT_NOT_NULL(mat->data);

    scnn_mat_free(&mat);
}

TEST(scnn_mat, alloc_shape_3d)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 3, 28, 28 } });

    TEST_ASSERT_EQUAL(1, mat->shape.d[0]);
    TEST_ASSERT_EQUAL(3, mat->shape.d[1]);
    TEST_ASSERT_EQUAL(28, mat->shape.d[2]);
    TEST_ASSERT_EQUAL(28, mat->shape.d[3]);

    TEST_ASSERT_EQUAL((3 * 28 * 28), mat->size);

    TEST_ASSERT_NOT_NULL(mat->data);

    scnn_mat_free(&mat);
}

TEST(scnn_mat, alloc_fail_with_zero)
{
    TEST_ASSERT_NULL(scnn_mat_alloc((scnn_shape){ .d = { 0 } }));
}

TEST(scnn_mat, alloc_fail_with_negative)
{
    TEST_ASSERT_NULL(scnn_mat_alloc((scnn_shape){ .d = { -1 } }));
}

TEST(scnn_mat, alloc_fail_with_invalid_shape_2d)
{
    TEST_ASSERT_NULL(scnn_mat_alloc((scnn_shape){ .d = { 0, 1 } }));
}

TEST(scnn_mat, alloc_fail_with_invalid_shape_3d)
{
    TEST_ASSERT_NULL(scnn_mat_alloc((scnn_shape){ .d = { 1, 0, 1 }}));
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

/*TEST(scnn_mat, init_and_free)
{
    scnn_mat *mat = scnn_mat_alloc();

    TEST_ASSERT_EQUAL_PTR(mat, scnn_mat_init(mat, 1, 3, 480, 640));

    TEST_ASSERT_EQUAL(1, mat->n);
    TEST_ASSERT_EQUAL(3, mat->c);
    TEST_ASSERT_EQUAL(480, mat->h);
    TEST_ASSERT_EQUAL(640, mat->w);

    TEST_ASSERT_EQUAL((1 * 3 * 480 * 640), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);

    scnn_mat_free(&mat);

    TEST_ASSERT_NULL(mat);
}*/

TEST(scnn_mat, fill)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 1, 3, 480, 640 } });

    TEST_ASSERT_EQUAL_PTR(mat, scnn_mat_fill(mat, 1));

    for (int i = 0; i < mat->size; i++) {
        TEST_ASSERT_EQUAL(1, mat->data[i]);
    }

    scnn_mat_free(&mat);
}

TEST(scnn_mat, fill_fail_null)
{
    TEST_ASSERT_NULL(scnn_mat_fill(NULL, 1));
}

/*TEST(scnn_mat, fill_fail_not_initialized)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 1, 10, 1, 1 } });

    TEST_ASSERT_NULL(scnn_mat_fill(mat, 1));

    scnn_mat_free(&mat);
}*/

TEST(scnn_mat, copy_from_array)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 1, 10, 1, 1 } });

    float array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_NOT_NULL(scnn_mat_copy_from_array(mat, array, 10));

    scnn_mat_free(&mat);
}

TEST(scnn_mat, copy_from_array_fail_invalid_size)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 1, 10, 1, 1 } });

    float array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_NULL(scnn_mat_copy_from_array(mat, array, 0));

    scnn_mat_free(&mat);
}

TEST(scnn_mat, copy_from_array_fail_mat_null)
{
    float array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_NULL(scnn_mat_copy_from_array(NULL, array, 10));
}

TEST(scnn_mat, copy_from_array_fail_array_null)
{
    scnn_mat *mat = scnn_mat_alloc((scnn_shape){ .d = { 1, 10, 1, 1 } });

    TEST_ASSERT_NULL(scnn_mat_copy_from_array(mat, NULL, 10));
}
