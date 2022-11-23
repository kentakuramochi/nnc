/**
 * @file test_scnn_softmax_runner.c
 * @brief Test runner of scnn_softmax.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_softmax)
{
    RUN_TEST_CASE(scnn_softmax, allocate_softmax_layer);

    RUN_TEST_CASE(scnn_softmax, initialize);
    RUN_TEST_CASE(scnn_softmax, cannot_initialize_with_NULL);
    RUN_TEST_CASE(scnn_softmax, cannot_initialize_without_in_shape);
    RUN_TEST_CASE(scnn_softmax, cannot_initialize_with_invalid_in_shape);

    RUN_TEST_CASE(scnn_softmax, forward);
    RUN_TEST_CASE(scnn_softmax, forward_with_xy_dim);
    RUN_TEST_CASE(scnn_softmax, forward_with_batch_dim);
    RUN_TEST_CASE(scnn_softmax, forward_fails_when_x_is_NULL);
    RUN_TEST_CASE(scnn_softmax, forward_fails_when_layer_is_NULL);

    RUN_TEST_CASE(scnn_softmax, backward);
    RUN_TEST_CASE(scnn_softmax, backward_with_xy_dim);
    RUN_TEST_CASE(scnn_softmax, backward_with_batch_dim);
    RUN_TEST_CASE(scnn_softmax, backward_fails_when_dy_is_NULL);
    RUN_TEST_CASE(scnn_softmax, backward_fails_when_layer_is_NULL);
}
