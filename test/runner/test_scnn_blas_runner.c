/**
 * @file test_scnn_blas_runner.c
 * @brief Test runner of scnn_blas.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_blas)
{
    RUN_TEST_CASE(scnn_blas, scopy);
    RUN_TEST_CASE(scnn_blas, scopy_incx_2);
    RUN_TEST_CASE(scnn_blas, scopy_rev_x);
    RUN_TEST_CASE(scnn_blas, scopy_rev_x_2);
    RUN_TEST_CASE(scnn_blas, scopy_incy_2);
    RUN_TEST_CASE(scnn_blas, scopy_rev_y);
    RUN_TEST_CASE(scnn_blas, scopy_rev_y_2);
    RUN_TEST_CASE(scnn_blas, scopy_fail_x_null);
    RUN_TEST_CASE(scnn_blas, scopy_fail_y_null);
    RUN_TEST_CASE(scnn_blas, scopy_fail_invalid_n);
    RUN_TEST_CASE(scnn_blas, scopy_fail_invalid_incx);
    RUN_TEST_CASE(scnn_blas, scopy_fail_invalid_incy);

    RUN_TEST_CASE(scnn_blas, sdot);
    RUN_TEST_CASE(scnn_blas, sdot_incx_2);
    RUN_TEST_CASE(scnn_blas, sdot_incy_2);
    RUN_TEST_CASE(scnn_blas, sdot_rev_x);
    RUN_TEST_CASE(scnn_blas, sdot_rev_y);
    RUN_TEST_CASE(scnn_blas, sdot_rev_x_2);
    RUN_TEST_CASE(scnn_blas, sdot_rev_y_2);
    RUN_TEST_CASE(scnn_blas, sdot_fail_invalid_n);
    RUN_TEST_CASE(scnn_blas, sdot_fail_x_null);
    RUN_TEST_CASE(scnn_blas, sdot_fail_y_null);
    RUN_TEST_CASE(scnn_blas, sdot_fail_invalid_incx);
    RUN_TEST_CASE(scnn_blas, sdot_fail_invalid_incy);

    RUN_TEST_CASE(scnn_blas, snrm2);
    RUN_TEST_CASE(scnn_blas, snrm2_incx_2);
    RUN_TEST_CASE(scnn_blas, snrm2_rev);
    RUN_TEST_CASE(scnn_blas, snrm2_rev_2);
    RUN_TEST_CASE(scnn_blas, snrm2_fail_invalid_n);
    RUN_TEST_CASE(scnn_blas, snrm2_fail_x_null);
    RUN_TEST_CASE(scnn_blas, snrm2_fail_invalid_incx);

    RUN_TEST_CASE(scnn_blas, saxpy);
    RUN_TEST_CASE(scnn_blas, saxpy_alpha_2);
    RUN_TEST_CASE(scnn_blas, saxpy_incx_2);
    RUN_TEST_CASE(scnn_blas, saxpy_incy_2);
    RUN_TEST_CASE(scnn_blas, saxpy_rev_x);
    RUN_TEST_CASE(scnn_blas, saxpy_rev_y);
    RUN_TEST_CASE(scnn_blas, saxpy_rev_x_2);
    RUN_TEST_CASE(scnn_blas, saxpy_rev_y_2);
    RUN_TEST_CASE(scnn_blas, saxpy_fail_x_null);
    RUN_TEST_CASE(scnn_blas, saxpy_fail_y_null);
    RUN_TEST_CASE(scnn_blas, saxpy_fail_invalid_n);
    RUN_TEST_CASE(scnn_blas, saxpy_fail_invalid_incx);
    RUN_TEST_CASE(scnn_blas, saxpy_fail_invalid_incy);

    RUN_TEST_CASE(scnn_blas, sgemv);
    RUN_TEST_CASE(scnn_blas, sgemv_trans);
    RUN_TEST_CASE(scnn_blas, sgemv_alpha_2);
    RUN_TEST_CASE(scnn_blas, sgemv_beta_2);
    RUN_TEST_CASE(scnn_blas, sgemv_incx_2);
    RUN_TEST_CASE(scnn_blas, sgemv_incx_rev);
    RUN_TEST_CASE(scnn_blas, sgemv_incx_rev_2);
    RUN_TEST_CASE(scnn_blas, sgemv_incy_2);
    RUN_TEST_CASE(scnn_blas, sgemv_incy_rev);
    RUN_TEST_CASE(scnn_blas, sgemv_incy_rev_2);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_a_null);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_x_null);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_c_null);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_invalid_m);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_invalid_n);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_invalid_lda);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_invalid_incx);
    RUN_TEST_CASE(scnn_blas, sgemv_fail_invalid_incy);

    RUN_TEST_CASE(scnn_blas, sgemm_no_trans);
    RUN_TEST_CASE(scnn_blas, sgemm_trans_b);
    //RUN_TEST_CASE(scnn_blas, sgemm_trans_a);
    //RUN_TEST_CASE(scnn_blas, sgemm_trans_ab);
    RUN_TEST_CASE(scnn_blas, sgemm_alpha_2);
    RUN_TEST_CASE(scnn_blas, sgemm_beta_2);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_a_null);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_b_null);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_c_null);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_n);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_m);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_k);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_lda);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_ldb);
    RUN_TEST_CASE(scnn_blas, sgemm_fail_invalid_ldc);
}
