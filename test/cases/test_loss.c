/**
 * @file test_loss.c
 * @brief Unit tests of loss.c
 *
 */
#include "loss.h"

#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

void test_mse_loss(void) {
    float loss = mse_loss(
        FLOAT_ARRAY(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0),
        FLOAT_ARRAY(0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        10
    );

    TEST_ASSERT_EQUAL_FLOAT(0.0195f, loss);
}

void test_se_loss(void) {
    float loss = se_loss(
        FLOAT_ARRAY(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0),
        FLOAT_ARRAY(0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        10
    );

    TEST_ASSERT_EQUAL_FLOAT(0.0975f, loss);
}
