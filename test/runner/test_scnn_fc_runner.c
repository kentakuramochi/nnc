/**
 * @file test_scnn_fc_runner.c
 * @brief Test runner of scnn_fc.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_fc)
{
    RUN_TEST_CASE(scnn_fc, alloc_and_free);

    RUN_TEST_CASE(scnn_fc, alloc_fail_invalid_param_in);
    RUN_TEST_CASE(scnn_fc, alloc_fail_invalid_param_out);
}
