/**
 * @file test_scnn_mat_runner.c
 * @brief Test runner of scnn_mat.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_mat)
{
    RUN_TEST_CASE(scnn_mat, alloc_and_free);

    RUN_TEST_CASE(scnn_mat, free_to_null);
    RUN_TEST_CASE(scnn_mat, free_to_ptr_to_null);

    RUN_TEST_CASE(scnn_mat, init_and_free);

    RUN_TEST_CASE(scnn_mat, init_fail_n_zero);
    RUN_TEST_CASE(scnn_mat, init_fail_c_zero);
    RUN_TEST_CASE(scnn_mat, init_fail_h_zero);
    RUN_TEST_CASE(scnn_mat, init_fail_w_zero);

    RUN_TEST_CASE(scnn_mat, init_fail_n_negative);
    RUN_TEST_CASE(scnn_mat, init_fail_c_negative);
    RUN_TEST_CASE(scnn_mat, init_fail_h_negative);
    RUN_TEST_CASE(scnn_mat, init_fail_w_negative);

    RUN_TEST_CASE(scnn_mat, fill);
    RUN_TEST_CASE(scnn_mat, fill_fail_null);
    RUN_TEST_CASE(scnn_mat, fill_fail_not_initialized);
}
