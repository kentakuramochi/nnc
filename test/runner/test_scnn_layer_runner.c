/**
 * @file test_scnn_layer_runner.c
 * @brief Test runner of scnn_layer.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_layer)
{
    RUN_TEST_CASE(scnn_layer, allocate_layer);

    RUN_TEST_CASE(scnn_layer, free_to_NULL_does_no_harm);
    RUN_TEST_CASE(scnn_layer, double_free_is_avoided);
}
