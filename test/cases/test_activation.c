/**
 * @file test_activation.c
 * @brief Unite tests of activation.c
 *
 */

#include "activation.h"

#include "unity.h"

void setUp(void)
{
}

void tearDown(void)
{
}

void test_sigmoid(void)
{
    float x[] = {
        -1, 0, 1
    };

    float y[3];

    float answer[] = {
        0.268941, 0.5, 0.731059
    };

    TEST_ASSERT_EQUAL_PTR(y, sigmoid(x, y, 3));

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, y, 3);
}
