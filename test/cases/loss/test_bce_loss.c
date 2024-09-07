/**
 * @file test_bce_loss.c
 * @brief Unit tests of bce_loss.c
 *
 */
#include "bce_loss.h"

#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void test_bce_loss(void) {
    LossFunc loss_func = bce_loss();

    float y[] = { 0.7, 0.3 };
    float t[] = { 1, 0 };

    TEST_ASSERT_EQUAL_FLOAT(0.356675, loss_func.forward(y, t, 2));
}

void test_bce_loss_backward(void) {
    LossFunc loss_func = bce_loss();

    float y[] = { 0.7, 0.3 };
    float t[] = { 1, 0 };

    float diff[2];
    loss_func.backward(diff, y, t, 2);

    float answer[] = { -0.714285, 0.714285 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, diff, 2);
}
