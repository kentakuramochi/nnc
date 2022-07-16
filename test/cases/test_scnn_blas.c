/**
 * @file test_scnn_blas.c
 * @brief Unit tests of scnn_blas.c
 * 
 */
#include "scnn_blas.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_blas);

TEST_SETUP(scnn_blas)
{}

TEST_TEAR_DOWN(scnn_blas)
{}

TEST(scnn_blas, sdot)
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

TEST(scnn_blas, sdot_incx_2)
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

TEST(scnn_blas, sdot_incy_2)
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

TEST(scnn_blas, sdot_rev_x)
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

TEST(scnn_blas, sdot_rev_y)
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

TEST(scnn_blas, sdot_rev_x_2)
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

TEST(scnn_blas, sdot_rev_y_2)
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

TEST(scnn_blas, sdot_fail_invalid_n)
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

TEST(scnn_blas, sdot_fail_x_null)
{
    int n = 10;
    float y[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_sdot(n, NULL, 1, y, 1));
}

TEST(scnn_blas, sdot_fail_y_null)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_sdot(n, x, 1, NULL, 1));
}

TEST(scnn_blas, sdot_fail_invalid_incx)
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

TEST(scnn_blas, sdot_fail_invalid_incy)
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

TEST(scnn_blas, snrm2)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(16.8819, scnn_snrm2(n, x, 1));
}

TEST(scnn_blas, snrm2_incx_2)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    };

    TEST_ASSERT_EQUAL_FLOAT(33.7639, scnn_snrm2(n, x, 2));
}

TEST(scnn_blas, snrm2_rev)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(16.8819, scnn_snrm2(n, x, -1));
}

TEST(scnn_blas, snrm2_rev_2)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    };

    TEST_ASSERT_EQUAL_FLOAT(36.4692, scnn_snrm2(n, x, -2));
}

TEST(scnn_blas, snrm2_fail_invalid_n)
{
    int n = 0;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_snrm2(n, x, 1));
}

TEST(scnn_blas, snrm2_fail_x_null)
{
    int n = 10;

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_snrm2(n, NULL, 1));
}

TEST(scnn_blas, snrm2_fail_invalid_incx)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    TEST_ASSERT_EQUAL_FLOAT(0, scnn_snrm2(n, x, 0));
}

TEST(scnn_blas, saxpy)
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

TEST(scnn_blas, saxpy_alpha_2)
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

TEST(scnn_blas, saxpy_incx_2)
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

TEST(scnn_blas, saxpy_incy_2)
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

TEST(scnn_blas, saxpy_rev_x)
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

TEST(scnn_blas, saxpy_rev_y)
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

TEST(scnn_blas, saxpy_rev_x_2)
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

TEST(scnn_blas, saxpy_rev_y_2)
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

TEST(scnn_blas, saxpy_fail_x_null)
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

TEST(scnn_blas, saxpy_fail_y_null)
{
    int n = 10;
    float x[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    };

    scnn_saxpy(n, 1.0, x, 1, NULL, 1);
}

TEST(scnn_blas, saxpy_fail_invalid_n)
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

TEST(scnn_blas, saxpy_fail_invalid_incx)
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

TEST(scnn_blas, saxpy_fail_invalid_incy)
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

TEST(scnn_blas, sgemv)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_trans)
{
    int lda = 2;
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
        3, 2,
        1.0, a, lda,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_alpha_2)
{
    int lda = 3;
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
        2.0, a, lda,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        0.48,
        3.68
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_beta_2)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 1,
        2.0, y, 1
    );

    float answer[] = {
        -1.26,
        3.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_incx_2)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 2,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_incx_rev)
{
    int lda = 3;
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
        1.0, a, lda,
        x, -1,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_incx_rev_2)
{
    int lda = 3;
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
        1.0, a, lda,
        x, -2,
        1.0, y, 1
    );

    float answer[] = {
        -0.26,
        2.34
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_incy_2)
{
    int lda = 3;
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
        1.0, a, lda,
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

TEST(scnn_blas, sgemv_incy_rev)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 1,
        1.0, y, -1
    );

    float answer[] = {
        2.34,
        -0.26
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_incy_rev_2)
{
    int lda = 3;
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
        1.0, a, lda,
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

TEST(scnn_blas, sgemv_fail_a_null)
{
    int lda = 3;

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
        1.0, NULL, lda,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_fail_x_null)
{
    int lda = 3;
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
        1.0, a, lda,
        NULL, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_fail_c_null)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 1,
        1.0, NULL, 1
    );
}

TEST(scnn_blas, sgemv_fail_invalid_m)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_fail_invalid_n)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_fail_invalid_lda)
{
    int lda = 0;
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
        1.0, a, lda,
        x, 1,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_fail_invalid_incx)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 0,
        1.0, y, 1
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemv_fail_invalid_incy)
{
    int lda = 3;
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
        1.0, a, lda,
        x, 1,
        1.0, y, 0
    );

    float answer[] = {
        -1,
        1
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 2);
}

TEST(scnn_blas, sgemm_no_trans)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_trans_b)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 3;
    float b[2 * 3] = {
        1011, 1021, 1031,
        1012, 1022, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_TRANS,
        2, 2, 3,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_trans_a)
{
    int lda = 2;
    float a[3 * 2] = {
        0.11, 0.21,
        0.12, 0.22,
        0.13, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_trans_ab)
{
    int lda = 2;
    float a[3 * 2] = {
        0.11, 0.21,
        0.12, 0.22,
        0.13, 0.23
    };

    int ldb = 3;
    float b[2 * 3] = {
        1011, 1021, 1031,
        1012, 1022, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_TRANS,
        SCNN_BLAS_TRANS,
        2, 2, 3,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        368.76, 370.12,
        677.06, 678.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_alpha_2)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        2.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        736.52, 738.24,
        1351.12, 1353.44
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_beta_2)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 3,
        1.0, a, lda,
        b, ldb,
        2.0, c, ldc
    );

    float answer[] = {
        369.76, 372.12,
        680.06, 682.72
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_a_null)
{
    int lda = 3;

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, NULL, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_b_null)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, a, lda,
        NULL, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_c_null)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, a, lda,
        b, ldb,
        1.0, NULL, ldc
    );
}

TEST(scnn_blas, sgemm_fail_invalid_m)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        0, 2, 3,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_invalid_n)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 0, 3,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_invalid_k)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_invalid_lda)
{
    int lda = 0;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_invalid_ldb)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 0;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 2;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}

TEST(scnn_blas, sgemm_fail_invalid_ldc)
{
    int lda = 3;
    float a[2 * 3] = {
        0.11, 0.12, 0.13,
        0.21, 0.22, 0.23
    };

    int ldb = 2;
    float b[3 * 2] = {
        1011, 1012,
        1021, 1022,
        1031, 1032
    };

    int ldc = 0;
    float c[2 * 2] = {
        1, 2,
        3, 4
    };

    scnn_sgemm(SCNN_BLAS_NO_TRANS,
        SCNN_BLAS_NO_TRANS,
        2, 2, 0,
        1.0, a, lda,
        b, ldb,
        1.0, c, ldc
    );

    float answer[] = {
        1, 2,
        3, 4
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, c, (2 * 2));
}
