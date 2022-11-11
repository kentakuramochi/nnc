/**
 * @file test_scnn_mat_runner.c
 * @brief Test runner of scnn_mat.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_mat)
{
    RUN_TEST_CASE(scnn_mat, allocate_with_1d_shape);
    RUN_TEST_CASE(scnn_mat, allocate_with_2d_shape);
    RUN_TEST_CASE(scnn_mat, allocate_with_3d_shape);
    RUN_TEST_CASE(scnn_mat, allocate_with_4d_shape);

    RUN_TEST_CASE(scnn_mat, free_to_NULL_does_no_harm);
    RUN_TEST_CASE(scnn_mat, double_free_is_avoided);

    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_shape_is_NULL);

    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_all_dims_are_0);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_1st_dim_is_0);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_2nd_dim_is_0);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_3rd_dim_is_0);
    RUN_TEST_CASE(scnn_mat, allocate_with_truncated_shape_when_last_dim_is_0);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_1st_dim_is_negative);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_2nd_dim_is_negative);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_3rd_dim_is_negative);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_when_4th_dim_is_negative);

    RUN_TEST_CASE(scnn_mat, fill_with_1);
    RUN_TEST_CASE(scnn_mat, cannot_fill_when_mat_is_not_allocated);

    RUN_TEST_CASE(scnn_mat, allocate_zeros);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_zeros_when_allocation_is_failed);

    RUN_TEST_CASE(scnn_mat, allocate_random);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_random_when_allocation_is_failed);

    RUN_TEST_CASE(scnn_mat, allocate_random_norm);
    RUN_TEST_CASE(scnn_mat, allocate_random_norm_mean_1_std_1);
    RUN_TEST_CASE(scnn_mat, allocate_random_norm_mean_0_std_2);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_random_norm_when_allocation_is_failed);

    RUN_TEST_CASE(scnn_mat, allocate_from_array);
    RUN_TEST_CASE(scnn_mat, allocate_from_same_array_with_different_shape);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_from_array_when_shape_is_NULL);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_from_array_with_unmatched_size);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_from_array_with_unmatched_shape);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_from_array_with_invalid_shape);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_from_array_when_size_is_0);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_from_array_when_size_is_negative);
    RUN_TEST_CASE(scnn_mat, cannot_allocate_from_array_when_array_is_NULL);
}
