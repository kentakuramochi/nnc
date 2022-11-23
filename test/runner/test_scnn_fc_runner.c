/**
 * @file test_scnn_fc_runner.c
 * @brief Test runner of scnn_fc.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_fc)
{
    RUN_TEST_CASE(scnn_fc, allocate_fc_layer);

    RUN_TEST_CASE(scnn_fc, initialize);
    RUN_TEST_CASE(scnn_fc, cannot_initialize_with_NULL);
    RUN_TEST_CASE(scnn_fc, cannot_initialize_without_in_shape);
    RUN_TEST_CASE(scnn_fc, cannot_initialize_with_invalid_in_shape);
    RUN_TEST_CASE(scnn_fc, cannot_initialize_with_invalid_out);

    RUN_TEST_CASE(scnn_fc, forward);
    RUN_TEST_CASE(scnn_fc, forward_with_batch_dim);
    RUN_TEST_CASE(scnn_fc, forward_fails_when_x_is_NULL);
    RUN_TEST_CASE(scnn_fc, forward_fails_when_layer_is_NULL);

    RUN_TEST_CASE(scnn_fc, backward);
    RUN_TEST_CASE(scnn_fc, backward_with_batch_dim);
    RUN_TEST_CASE(scnn_fc, backward_fails_when_dy_is_NULL);
    RUN_TEST_CASE(scnn_fc, backward_fails_when_layer_is_NULL);
}
