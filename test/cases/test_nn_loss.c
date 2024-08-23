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

void test_loss(void) {
    float y[] = { 1, 0 };
    float t[] = { 0, 0 };

    TEST_ASSERT_EQUAL_FLOAT(0.5, nn_loss(NN_LOSS_TYPE_MSE, y, t, 2));
}

void test_loss_backward(void) {
    float y[] = { 1, 0, 0 };
    float t[] = { 0, 0, 0 };
    float dy[3];

    nn_loss_backward(dy, NN_LOSS_TYPE_MSE, y, t, 3);

    // Test with MSE loss
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(TEST_UTIL_FLOAT_ARRAY(0.6666666, 0, 0), dy, 3);
}

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
