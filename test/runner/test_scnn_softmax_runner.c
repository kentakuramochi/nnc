/**
 * @file test_scnn_softmax_runner.c
 * @brief Test runner of scnn_softmax.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_softmax)
{
    RUN_TEST_CASE(scnn_softmax, alloc_and_free);

    RUN_TEST_CASE(scnn_softmax, alloc_fail_invalid_param_in);

    RUN_TEST_CASE(scnn_softmax, set_size);

    RUN_TEST_CASE(scnn_softmax, set_size_fail_invalid_n);
    RUN_TEST_CASE(scnn_softmax, set_size_fail_invalid_c);
    RUN_TEST_CASE(scnn_softmax, set_size_fail_invalid_h);
    RUN_TEST_CASE(scnn_softmax, set_size_fail_invalid_w);
    RUN_TEST_CASE(scnn_softmax, set_size_fail_invalid_in_size);

    RUN_TEST_CASE(scnn_softmax, forward);
    RUN_TEST_CASE(scnn_softmax, forward_fail_x_is_null);
    RUN_TEST_CASE(scnn_softmax, forward_fail_layer_is_null);

    RUN_TEST_CASE(scnn_softmax, backward);
    RUN_TEST_CASE(scnn_softmax, backward_fail_dy_is_null);
    RUN_TEST_CASE(scnn_softmax, backward_fail_layer_is_null);
}
