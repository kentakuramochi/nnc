/**
 * @file test_scnn_mat.c
 * @brief Unit tests of scnn_mat.c
 *
 */
#include <stdlib.h>

#include "scnn_mat.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_mat);

scnn_mat *mat;

TEST_SETUP(scnn_mat)
{
    mat = NULL;
}

TEST_TEAR_DOWN(scnn_mat)
{
    scnn_mat_free(&mat);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, allocate_with_1d_shape)
{
    mat = scnn_mat_alloc(scnn_shape(32));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->shape[0]);
    TEST_ASSERT_EQUAL(1, mat->shape[1]);
    TEST_ASSERT_EQUAL(1, mat->shape[2]);
    TEST_ASSERT_EQUAL(32, mat->shape[3]);

    TEST_ASSERT_EQUAL(32, mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);
}

TEST(scnn_mat, allocate_with_2d_shape)
{
    mat = scnn_mat_alloc(scnn_shape(28, 32));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->shape[0]);
    TEST_ASSERT_EQUAL(1, mat->shape[1]);
    TEST_ASSERT_EQUAL(28, mat->shape[2]);
    TEST_ASSERT_EQUAL(32, mat->shape[3]);

    TEST_ASSERT_EQUAL((28 * 32), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);
}

TEST(scnn_mat, allocate_with_3d_shape)
{
    mat = scnn_mat_alloc(scnn_shape(3, 28, 32));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->shape[0]);
    TEST_ASSERT_EQUAL(3, mat->shape[1]);
    TEST_ASSERT_EQUAL(28, mat->shape[2]);
    TEST_ASSERT_EQUAL(32, mat->shape[3]);

    TEST_ASSERT_EQUAL((3 * 28 * 32), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);
}

TEST(scnn_mat, allocate_with_4d_shape)
{
    mat = scnn_mat_alloc(scnn_shape(10, 3, 28, 32));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(10, mat->shape[0]);
    TEST_ASSERT_EQUAL(3, mat->shape[1]);
    TEST_ASSERT_EQUAL(28, mat->shape[2]);
    TEST_ASSERT_EQUAL(32, mat->shape[3]);

    TEST_ASSERT_EQUAL((10 * 3 * 28 * 32), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);
}

TEST(scnn_mat, free_to_NULL_does_no_harm)
{
    scnn_mat_free(NULL);

    // also freeing pointer to NULL is tested by
    // pair of TEST_SETUP and TEST_TEAR_DOWN
    // *mat = NULL;
    // scnn_mat_free(&mat);
}

TEST(scnn_mat, double_free_is_avoided)
{
    mat = scnn_mat_alloc(scnn_shape(10));

    scnn_mat_free(&mat);

    TEST_ASSERT_NULL(mat);

    // 2nd freeing is done in TEST_TEAR_DOWN
}

TEST(scnn_mat, cannot_allocate_when_shape_is_NULL)
{
    mat = scnn_mat_alloc(NULL);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_when_all_dims_are_0)
{
    mat = scnn_mat_alloc(scnn_shape(0));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_when_1st_dim_is_0)
{
    mat = scnn_mat_alloc(scnn_shape(0, 3, 28, 32));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_when_2nd_dim_is_0)
{
    mat = scnn_mat_alloc(scnn_shape(10, 0, 28, 32));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_when_3rd_dim_is_0)
{
    mat = scnn_mat_alloc(scnn_shape(10, 3, 0, 32));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, allocate_with_truncated_shape_when_last_dim_is_0)
{
    mat = scnn_mat_alloc(scnn_shape(10, 3, 28, 0));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->shape[0]);
    TEST_ASSERT_EQUAL(10, mat->shape[1]);
    TEST_ASSERT_EQUAL(3, mat->shape[2]);
    TEST_ASSERT_EQUAL(28, mat->shape[3]);

    TEST_ASSERT_EQUAL((10 * 3 * 28), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_NOT_NULL(mat->data);
}

TEST(scnn_mat, cannot_allocate_when_1st_dim_is_negative)
{
    mat = scnn_mat_alloc(scnn_shape(-1, 3, 28, 32));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_when_2nd_dim_is_negative)
{
    mat = scnn_mat_alloc(scnn_shape(10, -1, 28, 32));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_when_3rd_dim_is_negative)
{
    mat = scnn_mat_alloc(scnn_shape(10, 3, -1, 32));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_when_4th_dim_is_negative)
{
    mat = scnn_mat_alloc(scnn_shape(10, 3, 28, -1));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, fill_with_1)
{
    mat = scnn_mat_alloc(scnn_shape(10));

    TEST_ASSERT_EQUAL_PTR(mat, scnn_mat_fill(mat, 1));

    TEST_ASSERT_EACH_EQUAL_FLOAT(1, mat->data, mat->size);
}

TEST(scnn_mat, cannot_fill_when_mat_is_not_allocated)
{
    TEST_ASSERT_NULL(scnn_mat_fill(mat, 1));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, allocate_zeros)
{
    mat = scnn_mat_zeros(scnn_shape(10, 3, 28, 32));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(10, mat->shape[0]);
    TEST_ASSERT_EQUAL(3, mat->shape[1]);
    TEST_ASSERT_EQUAL(28, mat->shape[2]);
    TEST_ASSERT_EQUAL(32, mat->shape[3]);

    TEST_ASSERT_EQUAL((10 * 3 * 28 * 32), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, mat->data, mat->size);
}

TEST(scnn_mat, cannot_allocate_zeros_when_allocation_is_failed)
{
    mat = scnn_mat_zeros(scnn_shape(0));

    TEST_ASSERT_NULL(mat);
}

// get array of random value with uniform distribution
static void get_rand_array(float *array, int size)
{
    for (int i = 0; i < size; i++) {
        array[i] = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    }
}

TEST(scnn_mat, allocate_random)
{
    srand(0);

    float rand_vals[10 * 3 * 28 * 32];
    get_rand_array(rand_vals, (10 * 3 * 28 * 32));

    srand(0);

    mat = scnn_mat_rand(scnn_shape(10, 3, 28, 32));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(10, mat->shape[0]);
    TEST_ASSERT_EQUAL(3, mat->shape[1]);
    TEST_ASSERT_EQUAL(28, mat->shape[2]);
    TEST_ASSERT_EQUAL(32, mat->shape[3]);

    TEST_ASSERT_EQUAL((10 * 3 * 28 * 32), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(rand_vals, mat->data, mat->size);
}

TEST(scnn_mat, cannot_allocate_random_when_allocation_is_failed)
{
    mat = scnn_mat_rand(scnn_shape(0));

    TEST_ASSERT_NULL(mat);
}

// get array of random value with normal distribution
static const float PI = 3.141592f;
static void get_randn_array(float *array, int size, float mean, float std)
{
    for (int i = 0; i < size; i++) {
        // Box-Muller's method
        float x = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float y = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float z = sqrtf(-2 * logf(x)) * cosf(2 * PI * y);

        array[i] = std * z + mean;
    }
}

TEST(scnn_mat, allocate_random_norm)
{
    srand(0);

    float mean = 0;
    float std = 1;

    float rand_vals[3 * 28 * 28];
    get_randn_array(rand_vals, (3 * 28 * 28), mean, std);

    srand(0);

    mat = scnn_mat_randn(scnn_shape(1, 3, 28, 28), mean, std);

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->shape[0]);
    TEST_ASSERT_EQUAL(3, mat->shape[1]);
    TEST_ASSERT_EQUAL(28, mat->shape[2]);
    TEST_ASSERT_EQUAL(28, mat->shape[3]);

    TEST_ASSERT_EQUAL((1 * 3 * 28 * 28), mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(rand_vals, mat->data, mat->size);
}

TEST(scnn_mat, allocate_random_norm_mean_1_std_1)
{
    srand(0);

    float mean = 1;
    float std = 1;

    float rand_vals[28];
    get_randn_array(rand_vals, 28, mean, std);

    srand(0);

    mat = scnn_mat_randn(scnn_shape(28), mean, std);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(rand_vals, mat->data, mat->size);
}

TEST(scnn_mat, allocate_random_norm_mean_0_std_2)
{
    srand(0);

    float mean = 0;
    float std = 2;

    float rand_vals[28];
    get_randn_array(rand_vals, 28, mean, std);

    srand(0);

    mat = scnn_mat_randn(scnn_shape(28), 0, 2);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(rand_vals, mat->data, mat->size);
}

TEST(scnn_mat, cannot_allocate_random_norm_when_allocation_is_failed)
{
    mat = scnn_mat_randn(scnn_shape(0), 0, 1);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, allocate_from_array)
{
    scnn_dtype array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    mat = scnn_mat_from_array(array, 10, scnn_shape(10));

    TEST_ASSERT_NOT_NULL(mat);

    TEST_ASSERT_EQUAL(1, mat->shape[0]);
    TEST_ASSERT_EQUAL(1, mat->shape[1]);
    TEST_ASSERT_EQUAL(1, mat->shape[2]);
    TEST_ASSERT_EQUAL(10, mat->shape[3]);

    TEST_ASSERT_EQUAL(10, mat->size);

    TEST_ASSERT_EQUAL(SCNN_MAT_ORDER_NCHW, mat->order);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(array, mat->data, mat->size);
}

TEST(scnn_mat, allocate_from_same_array_with_different_shape)
{
    scnn_dtype array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    mat = scnn_mat_from_array(array, 10, scnn_shape(10));

    scnn_mat *mat2 = scnn_mat_from_array(array, 10, scnn_shape(10, 1, 1, 1));

    TEST_ASSERT_NOT_NULL(mat2);

    TEST_ASSERT_EQUAL(1, mat->shape[0]);
    TEST_ASSERT_EQUAL(1, mat->shape[1]);
    TEST_ASSERT_EQUAL(1, mat->shape[2]);
    TEST_ASSERT_EQUAL(10, mat->shape[3]);

    TEST_ASSERT_EQUAL(10, mat2->shape[0]);
    TEST_ASSERT_EQUAL(1, mat2->shape[1]);
    TEST_ASSERT_EQUAL(1, mat2->shape[2]);
    TEST_ASSERT_EQUAL(1, mat2->shape[3]);

    TEST_ASSERT_EQUAL(mat->size, mat2->size);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(mat->data, mat2->data, mat->size);

    scnn_mat_free(&mat2);
}

TEST(scnn_mat, cannot_allocate_from_array_when_shape_is_NULL)
{
    scnn_dtype array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    mat = scnn_mat_from_array(array, 10, NULL);

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_from_array_with_unmatched_size)
{
    scnn_dtype array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    mat = scnn_mat_from_array(array, 1, scnn_shape(10));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_from_array_with_unmatched_shape)
{
    scnn_dtype array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    mat = scnn_mat_from_array(array, 10, scnn_shape(1));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_from_array_with_invalid_shape)
{
    scnn_dtype array[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    mat = scnn_mat_from_array(array, 10, scnn_shape(10, 0, 1));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_from_array_when_size_is_0)
{
    mat = scnn_mat_from_array(NULL, 0, scnn_shape(0));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_from_array_when_size_is_negative)
{
    mat = scnn_mat_from_array(NULL, -1, scnn_shape(10));

    TEST_ASSERT_NULL(mat);
}

TEST(scnn_mat, cannot_allocate_from_array_when_array_is_NULL)
{
    mat = scnn_mat_from_array(NULL, 10, scnn_shape(10));

    TEST_ASSERT_NULL(mat);
}
