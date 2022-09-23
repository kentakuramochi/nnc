/**
 * @file test_scnn_layer_runner.c
 * @brief Test runner of scnn_layer.c
 * 
 */
#include "unity_fixture.h"

TEST_GROUP_RUNNER(scnn_layer)
{
    RUN_TEST_CASE(scnn_layer, alloc_and_free);

    RUN_TEST_CASE(scnn_layer, free_to_null);
    RUN_TEST_CASE(scnn_layer, free_to_ptr_to_null);

    RUN_TEST_CASE(scnn_layer, set_shape_1d);
    RUN_TEST_CASE(scnn_layer, set_shape_2d);
    RUN_TEST_CASE(scnn_layer, set_shape_3d);
    RUN_TEST_CASE(scnn_layer, set_shape_4d);
}
