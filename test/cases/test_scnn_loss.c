/**
 * @file test_scnn_loss.c
 * @brief Unit tests of scnn_loss.c
 * 
 */
#include "scnn_loss.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_loss);

TEST_SETUP(scnn_loss)
{}

TEST_TEAR_DOWN(scnn_loss)
{}

TEST(scnn_loss, mean_squared_error)
{
    float x[10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    float t[10] = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

    float loss = scnn_mean_squared_error(x, t, 10);

    TEST_ASSERT_EQUAL_FLOAT(0.0975, loss);
}

TEST(scnn_loss, cross_entropy_loss)
{
    float x[10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    float t[10] = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0 };

    float loss = scnn_cross_entropy_loss(x, t, 10);

    TEST_ASSERT_EQUAL(6.447238, loss);
}
