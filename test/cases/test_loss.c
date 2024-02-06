/**
 * @file test_loss.c
 * @brief Unit tests of loss.c
 * 
 */

#include "loss.h"

#include "unity.h"

void setUp(void)
{
}

void tearDown(void)
{
}

void test_mse_loss(void)
{
    float y[10] = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };
    float t[10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };

    float loss = mse_loss(y, t, 10);

    TEST_ASSERT_EQUAL_FLOAT(0.0195f, loss);
}
