/**
 * @file test_nn_loss.c
 * @brief Unit tests of nn_loss.c
 *
 */
#include "nn_loss.h"

#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_mse_loss(void) {
    float loss = mse_loss(
        TEST_UTIL_FLOAT_ARRAY(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0),
        TEST_UTIL_FLOAT_ARRAY(0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        10
    );

    TEST_ASSERT_EQUAL_FLOAT(0.0195f, loss);
}

void test_binary_cross_entropy_loss(void) {
    float loss = binary_cross_entropy_loss(
        TEST_UTIL_FLOAT_ARRAY(0.7, 0.3),
        TEST_UTIL_FLOAT_ARRAY(1, 0),
        2
    );

    TEST_ASSERT_EQUAL_FLOAT(0.514573f, loss);
}
