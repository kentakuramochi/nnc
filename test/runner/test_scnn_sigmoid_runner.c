/**
 * @file test_scnn_sigmoid_runner.c
 * @brief Test runner of scnn_sigmoid.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_sigmoid)
{
    RUN_TEST_CASE(scnn_sigmoid, allocate_sigmoid_layer);

    RUN_TEST_CASE(scnn_sigmoid, initialize);
    RUN_TEST_CASE(scnn_sigmoid, cannot_initialize_with_NULL);
    RUN_TEST_CASE(scnn_sigmoid, cannot_initialize_with_invalid_in_shape);

    RUN_TEST_CASE(scnn_sigmoid, forward);
    RUN_TEST_CASE(scnn_sigmoid, forward_with_batch_dim);
    RUN_TEST_CASE(scnn_sigmoid, forward_fails_when_x_is_NULL);
    RUN_TEST_CASE(scnn_sigmoid, forward_fails_when_layer_is_NULL);

    RUN_TEST_CASE(scnn_sigmoid, backward);
    RUN_TEST_CASE(scnn_sigmoid, backward_with_batch_dim);
    RUN_TEST_CASE(scnn_sigmoid, backward_fails_when_dy_is_NULL);
    RUN_TEST_CASE(scnn_sigmoid, backward_fails_when_layer_is_NULL);
}
