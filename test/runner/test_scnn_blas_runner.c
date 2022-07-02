/**
 * @file test_scnn_blas_runner.c
 * @brief Test runner of scnn_blas.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_blas)
{
    RUN_TEST_CASE(scnn_blas, sgemm_no_trans);
    RUN_TEST_CASE(scnn_blas, sgemm_trans_b);
    RUN_TEST_CASE(scnn_blas, sgemm_trans_a);
    RUN_TEST_CASE(scnn_blas, sgemm_trans_ab);
    RUN_TEST_CASE(scnn_blas, sgemm_alpha_2);
    RUN_TEST_CASE(scnn_blas, sgemm_beta_2);

    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_n);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_m);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_k);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_lda);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_ldb);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_ldc);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_a_null);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_b_null);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_c_null);
}
