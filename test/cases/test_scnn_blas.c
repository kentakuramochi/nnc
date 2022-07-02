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
