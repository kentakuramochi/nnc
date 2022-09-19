/**
 * @file test_scnn_net_runner.c
 * @brief Test runner of scnn_net.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_net)
{
    RUN_TEST_CASE(scnn_net, alloc_and_free);

    RUN_TEST_CASE(scnn_net, append);
    RUN_TEST_CASE(scnn_net, append_2layers);
    RUN_TEST_CASE(scnn_net, append_3layers);

    RUN_TEST_CASE(scnn_net, append_net_is_null);
    RUN_TEST_CASE(scnn_net, append_layer_is_null);
    RUN_TEST_CASE(scnn_net, append_unmatched_size);
    RUN_TEST_CASE(scnn_net, append_over_max_size);

    RUN_TEST_CASE(scnn_net, forward);
    RUN_TEST_CASE(scnn_net, forward_net_is_null);
    RUN_TEST_CASE(scnn_net, forward_x_is_null);
    RUN_TEST_CASE(scnn_net, backward);
    RUN_TEST_CASE(scnn_net, backward_net_is_null);
    RUN_TEST_CASE(scnn_net, backward_t_is_null);
}
