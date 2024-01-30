/**
 * @file test_scnn_blas.c
 * @brief Unit tests of scnn_blas.c
 * 
 */
#include "scnn_blas.h"

#include "unity.h"

void setUp(void)
{}

void tearDown(void)
{}

void test_scopy(void)
{
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    float y[10] = { 0 };
    
    scnn_scopy(10, x, 1, y, 1);

    float answer[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 10);
}

void test_scopy_incx_2(void)
{
    float x[] = {
        0, 0, 1, 0, 2, 0, 3, 0, 4, 0,
        5, 0, 6, 0, 7, 0, 8, 0, 9, 0
    };

    float y[10] = { 0 };
    
    scnn_scopy(10, x, 2, y, 1);

    float answer[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 10);
}

void test_scopy_rev_x(void)
{
    float x[] = {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };

    float y[10] = { 0 };
    
    scnn_scopy(10, x, -1, y, 1);

    float answer[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 10);
}

void test_scopy_rev_x_2(void)
{
    float x[] = {
        0, 9, 0, 8, 0, 7, 0, 6, 0, 5,
        0, 4, 0, 3, 0, 2, 0, 1, 0, 0
    };

    float y[10] = { 0 };
    
    scnn_scopy(10, x, -2, y, 1);

    float answer[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 10);
}

void test_scopy_incy_2(void)
{
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    float y[20] = { 0 };
    
    scnn_scopy(10, x, 1, y, 2);

    float answer[] = {
        0, 0, 1, 0, 2, 0, 3, 0, 4, 0,
        5, 0, 6, 0, 7, 0, 8, 0, 9, 0
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 20);
}

void scopy_rev_y(void)
{
    float x[] = {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };

    float y[10] = { 0 };
    
    scnn_scopy(10, x, 1, y, -1);

    float answer[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 10);
}

void test_scopy_rev_y_2(void)
{
    float x[] = {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };

    float y[20] = { 0 };
    
    scnn_scopy(10, x, 1, y, -2);

    float answer[] = {
        0, 0, 0, 1, 0, 2, 0, 3, 0, 4,
        0, 5, 0, 6, 0, 7, 0, 8, 0, 9
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 20);
}

void test_scopy_fail_x_NULL(void)
{
    float y[10] = { 0 };
    
    scnn_scopy(10, NULL, 1, y, 1);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, y, 10);
}

void test_scopy_fail_y_NULL(void)
{
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    
    scnn_scopy(10, x, 1, NULL, 1);
}

void test_scopy_fail_invalid_n(void)
{
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    float y[10] = { 0 };
    
    scnn_scopy(0, x, 1, y, 1);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, y, 10);
}

void test_scopy_fail_invalid_incx(void)
{
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    float y[10] = { 0 };
    
    scnn_scopy(10, x, 0, y, 1);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, y, 10);
}

void test_scopy_fail_invalid_incy(void)
{
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    float y[10] = { 0 };
    
    scnn_scopy(10, x, 1, y, 0);

    TEST_ASSERT_EACH_EQUAL_FLOAT(0, y, 10);
}

void test_sdot(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(330, scnn_sdot(n, x, 1, y, 1));
}

void test_sdot_incx_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(660, scnn_sdot(n, x, 2, y, 1));
}

void test_sdot_incy_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    };

    TEST_ASSERT_EQUAL_FLOAT(615, scnn_sdot(n, x, 1, y, 2));
}

void test_sdot_rev_x(void)
{
    int n = 10;
    float x[] = {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(330, scnn_sdot(n, x, -1, y, 1));
}

void test_sdot_rev_y(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    };

    TEST_ASSERT_EQUAL_FLOAT(330, scnn_sdot(n, x, 1, y, -1));
}

void test_sdot_rev_x_2(void)
{
    int n = 10;
    float x[] = {
        19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(660, scnn_sdot(n, x, -2, y, 1));
}

void test_sdot_rev_y_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
        10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    };

    TEST_ASSERT_EQUAL_FLOAT(615, scnn_sdot(n, x, 1, y, -2));
}

void test_sdot_fail_invalid_n(void)
{
    int n = 0;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_sdot(n, x, 1, y, 1));
}

void test_sdot_fail_x_null(void)
{
    int n = 10;
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_sdot(n, NULL, 1, y, 1));
}

void test_sdot_fail_y_null(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_sdot(n, x, 1, NULL, 1));
}

void test_sdot_fail_invalid_incx(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_sdot(n, x, 0, y, 1));
}

void test_sdot_fail_invalid_incy(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_sdot(n, x, 1, y, 0));
}

void test_snrm2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(16.8819, scnn_snrm2(n, x, 1));
}

void test_snrm2_incx_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    };

    TEST_ASSERT_EQUAL_FLOAT(33.7639, scnn_snrm2(n, x, 2));
}

void test_snrm2_rev(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(16.8819, scnn_snrm2(n, x, -1));
}

void test_snrm2_rev_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    };

    TEST_ASSERT_EQUAL_FLOAT(36.4692, scnn_snrm2(n, x, -2));
}

void test_snrm2_fail_invalid_n(void)
{
    int n = 0;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_snrm2(n, x, 1));
}

void test_snrm2_fail_x_null(void)
{
    int n = 10;

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_snrm2(n, NULL, 1));
}

void test_snrm2_fail_invalid_incx(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_snrm2(n, x, 0));
}

void test_saxpy(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, x, 1, y, 1);

    float answer[] = {
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_alpha_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 2.0, x, 1, y, 1);

    float answer[] = {
        1, 4, 7, 10, 13, 16, 19, 22, 25, 28
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_incx_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, x, 2, y, 1);

    float answer[] = {
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_incy_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    scnn_saxpy(n, 1.0, x, 1, y, 2);

    float answer[] = {
        1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 10, 1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, (n * 2));
}

void test_saxpy_rev_x(void)
{
    int n = 10;
    float x[] = {
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, x, -1, y, 1);

    float answer[] = {
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_rev_y(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    };

    scnn_saxpy(n, 1.0, x, 1, y, -1);

    float answer[] = {
        19, 17, 15, 13, 11, 9, 7, 5, 3, 1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_rev_x_2(void)
{
    int n = 10;
    float x[] = {
        1, 9, 1, 8, 1, 7, 1, 6, 1, 5, 1, 4, 1, 3, 1, 2, 1, 1, 1, 0
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, x, -2, y, 1);

    float answer[] = {
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_rev_y_2(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 10, 1, 9, 1, 8, 1, 7, 1, 6, 1, 5, 1, 4, 1, 3, 1, 2, 1, 1
    };

    scnn_saxpy(n, 1.0, x, 1, y, -2);

    float answer[] = {
        1, 19, 1, 17, 1, 15, 1, 13, 1, 11, 1, 9, 1, 7, 1, 5, 1, 3, 1, 1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, (n * 2));
}

void test_saxpy_fail_x_null(void)
{
    int n = 10;
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, NULL, 1, y, 1);

    float answer[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_fail_y_null(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    scnn_saxpy(n, 1.0, x, 1, NULL, 1);
}

void test_saxpy_fail_invalid_n(void)
{
    int n = 0;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, x, 1, y, 1);

    float answer[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 10);
}

void test_saxpy_fail_invalid_incx(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, x, 0, y, 1);

    float answer[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_saxpy_fail_invalid_incy(void)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    scnn_saxpy(n, 1.0, x, 1, y, 0);

    float answer[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, n);
}

void test_sgemv(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_trans(void)
{
    float a[3 * 2] = {
        0.11, 0.21,
        0.12, 0.22,
        0.13, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_TRANS,
        2, 3,
        1.0, a, 2,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_alpha_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        2.0, a, 3,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        0.48,
        3.68
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_beta_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 1,
        2.0, y, 1
    );

    float answer[] = {
        -1.26,
        3.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_incx_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[] = {
        1,
        0,
        2,
        0,
        3,
        0
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 2,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_incx_rev(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        3,
        2,
        1
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, -1,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_incx_rev_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[] = {
        0,
        3,
        0,
        2,
        0,
        1
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, -2,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_incy_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[] = {
        -1,
        0,
        1,
        0
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 1,
        1.0, y, 2
    );

    float answer[] = {
        -0.26,
        0,
        2.34,
        0
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, (2 * 2));
}

void test_sgemv_incy_rev(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        1, 
        -1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 1,
        1.0, y, -1
    );

    float answer[] = {
        2.34,
        -0.26
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_incy_rev_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[] = {
        0,
        1,
        0,
        -1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 1,
        1.0, y, -2
    );

    float answer[] = {
        0,
        2.34,
        0,
        -0.26
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, (2 * 2));
}

void test_sgemv_fail_a_null(void)
{
    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, NULL, 3,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_fail_x_null(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        NULL, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_fail_c_null(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 1,
        1.0, NULL, 1
    );
}

void test_sgemv_fail_invalid_m(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        0, 3,
        1.0, a, 3,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_fail_invalid_n(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 0,
        1.0, a, 3,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_fail_invalid_lda(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 0,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_fail_invalid_incx(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1 
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 0,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemv_fail_invalid_incy(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float x[3] = {
        1,
        2,
        3
    };

    float y[2] = {
        -1,
        1
    };

    scnn_sgemv(SCNN_BLAS_NO_TRANS,
        2, 3,
        1.0, a, 3,
        x, 1,
        1.0, y, 0
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

void test_sgemm_no_trans(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 3,
        b, 2,
        1.0, c, 2 
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_trans_b(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[2 * 3] = {
        1011, 1021, 1031,
        1012, 1022, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_TRANS,
        2, 2, 3,
        1.0, a, 3,
        b, 3,
        1.0, c, 2
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_trans_a(void)
{
    float a[3 * 2] = {
        0.11, 0.21,
        0.12, 0.22,
        0.13, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 2,
        b, 2,
        1.0, c, 2
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_trans_ab(void)
{
    float a[3 * 2] = {
        0.11, 0.21,
        0.12, 0.22,
        0.13, 0.23
    };

    float b[2 * 3] = {
        1011, 1021, 1031,
        1012, 1022, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_TRANS,
        SCNN_BLAS_TRANS,
        2, 2, 3,
        1.0, a, 2,
        b, 3,
        1.0, c, 2 
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_alpha_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        2.0, a, 3,
        b, 2,
        1.0, c, 2
    );

    float answer[] = {
        736.52, 738.24,
        1351.12, 1353.44
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_beta_2(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 3,
        b, 2,
        2.0, c, 2
    );

    float answer[] = {
        369.76, 372.12,
        680.06, 682.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_a_null(void)
{
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, NULL, 3,
        b, 2,
        1.0, c, 2
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_b_null(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 3,
        NULL, 2,
        1.0, c, 2
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_c_null(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 3,
        b, 2,
        1.0, NULL, 2
    );
}

void test_sgemm_fail_invalid_m(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        0, 2, 3,
        1.0, a, 3,
        b, 2,
        1.0, c, 2
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_invalid_n(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 0, 3,
        1.0, a, 3,
        b, 2,
        1.0, c, 2
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_invalid_k(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, a, 3,
        b, 2,
        1.0, c, 2
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_invalid_lda(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 0,
        b, 2,
        1.0, c, 2
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_invalid_ldb(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 3,
        b, 0,
        1.0, c, 2
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

void test_sgemm_fail_invalid_ldc(void)
{
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, 2,
        b, 2,
        1.0, c, 0
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}
