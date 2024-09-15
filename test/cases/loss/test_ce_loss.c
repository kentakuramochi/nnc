/**
 * @file test_ce_loss.c
 * @brief Unit tests of ce_loss.c
 *
 */
#include "ce_loss.h"

#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void test_ce_loss(void) {
    LossFunc loss_func = ce_loss();

    float y[] = {
        0.7, 0.2, 0.1,
        0.2, 0.6, 0.2
    };

    float t[] = {
        1, 0, 0,
        0, 0, 1
    };

    // Result is based on PyTorch.torch.nn.NLLLoss()
    TEST_ASSERT_EQUAL_FLOAT(0.9830564, loss_func.forward(y, t, 2, 3));
}

void test_ce_loss_backward(void) {
    LossFunc loss_func = ce_loss();

    float y[] = {
        0.7, 0.2, 0.1,
        0.2, 0.6, 0.2
    };

    float t[] = {
        1, 0, 0,
        0, 0, 1
    };

    float diff[2 * 3];
    loss_func.backward(diff, y, t, 2, 3);

    // Result is based on PyTorch.torch.nn.NLLLoss()
    float answer[] = {
        -0.7142857, 0, 0,
        0, 0, -2.5000000
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, diff, (2 * 3));
}
