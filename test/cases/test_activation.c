/**
 * @file test_activation.c
 * @brief Unite tests of activation.c
 *
 */
#include "activation.h"

#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_sigmoid(void) {
    float y[3];

    TEST_ASSERT_EQUAL_PTR(
        y,
        sigmoid(FLOAT_ARRAY(-1, 0, 1), y, 3)
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(0.268941, 0.5, 0.731059),
        y,
        3
    );
}

void test_softmax(void) {
    float y[3];

    TEST_ASSERT_EQUAL_PTR(
        y,
        softmax(FLOAT_ARRAY(-1, 1, 4), y, 3)
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(0.00637746f, 0.04712342f, 0.94649912f),
        y,
        3
    );
}
