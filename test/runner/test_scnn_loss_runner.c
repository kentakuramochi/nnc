/**
 * @file test_scnn_loss_runner.c
 * @brief Test runner of scnn_loss.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_loss)
{
    RUN_TEST_CASE(scnn_loss, mean_squared_error);

    RUN_TEST_CASE(scnn_loss, cross_entropy_loss);
}
