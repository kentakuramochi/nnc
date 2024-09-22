/**
 * @file test_bce_loss.c
 * @brief Unit tests of bce_loss.c
 */
#include "bce_loss.h"

#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void test_bce_loss(void) {
    LossFunc loss_func = bce_loss();

    float y[] = { 0.7, 0.2, 0.1 };
    float t[] = { 1, 0, 0 };

    TEST_ASSERT_EQUAL_FLOAT(0.2283930, loss_func.forward(y, t, 3, 1));
}

void test_bce_loss_backward(void) {
    LossFunc loss_func = bce_loss();

    float y[] = { 0.7, 0.2, 0.1 };
    float t[] = { 1, 0, 0 };

    float grad[3];
    loss_func.backward(grad, y, t, 3, 1);

    float answer[] = { -0.4761905, 0.4166666, 0.3703704 };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, grad, 3);
}
