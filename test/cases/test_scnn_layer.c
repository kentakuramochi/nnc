/**
 * @file test_scnn_layer.c
 * @brief Unit tests of scnn_layer.c
 * 
 */
#include "scnn_layer.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_layer);

TEST_SETUP(scnn_layer)
{}

TEST_TEAR_DOWN(scnn_layer)
{}

TEST(scnn_layer, alloc_and_free)
{
    scnn_layer_params params = { .in_shape = { 1, 3, 28, 28 } };
    scnn_layer *layer = scnn_layer_alloc(params);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL_INT_ARRAY(params.in_shape, layer->params.in_shape, 4);

    TEST_ASSERT_EQUAL(0, layer->id);

    TEST_ASSERT_EQUAL(0, layer->prev_id);
    TEST_ASSERT_EQUAL(0, layer->next_id);

    TEST_ASSERT_EQUAL(NULL, layer->forward);
    TEST_ASSERT_EQUAL(NULL, layer->backward);

    TEST_ASSERT_EQUAL(NULL, layer->set_size);

    scnn_layer_free(&layer);

    TEST_ASSERT_NULL(layer);
}

TEST(scnn_layer, free_to_null)
{
    scnn_layer_free(NULL);
}

TEST(scnn_layer, free_to_ptr_to_null)
{
    scnn_layer *layer = NULL;

    scnn_layer_free(&layer);
}

TEST(scnn_layer, set_shape_1d)
{
    scnn_layer *layer = scnn_layer_alloc((scnn_layer_params){ .in_shape = { 28 } });

    TEST_ASSERT_EQUAL_INT(1, layer->params.in_shape[0]);
    TEST_ASSERT_EQUAL_INT(1, layer->params.in_shape[1]);
    TEST_ASSERT_EQUAL_INT(1, layer->params.in_shape[2]);
    TEST_ASSERT_EQUAL_INT(28, layer->params.in_shape[3]);

    scnn_layer_free(&layer);
}

TEST(scnn_layer, set_shape_2d)
{
    scnn_layer *layer = scnn_layer_alloc((scnn_layer_params){ .in_shape = { 28, 28 } });

    TEST_ASSERT_EQUAL_INT(1, layer->params.in_shape[0]);
    TEST_ASSERT_EQUAL_INT(1, layer->params.in_shape[1]);
    TEST_ASSERT_EQUAL_INT(28, layer->params.in_shape[2]);
    TEST_ASSERT_EQUAL_INT(28, layer->params.in_shape[3]);

    scnn_layer_free(&layer);
}

TEST(scnn_layer, set_shape_3d)
{
    scnn_layer *layer = scnn_layer_alloc((scnn_layer_params){ .in_shape = { 3, 28, 28 } });

    TEST_ASSERT_EQUAL_INT(1, layer->params.in_shape[0]);
    TEST_ASSERT_EQUAL_INT(3, layer->params.in_shape[1]);
    TEST_ASSERT_EQUAL_INT(28, layer->params.in_shape[2]);
    TEST_ASSERT_EQUAL_INT(28, layer->params.in_shape[3]);

    scnn_layer_free(&layer);
}

TEST(scnn_layer, set_shape_4d)
{
    scnn_layer *layer = scnn_layer_alloc((scnn_layer_params){ .in_shape = { 10, 3, 28, 28 } });

    TEST_ASSERT_EQUAL_INT(10, layer->params.in_shape[0]);
    TEST_ASSERT_EQUAL_INT(3, layer->params.in_shape[1]);
    TEST_ASSERT_EQUAL_INT(28, layer->params.in_shape[2]);
    TEST_ASSERT_EQUAL_INT(28, layer->params.in_shape[3]);

    scnn_layer_free(&layer);
}

TEST(scnn_layer, set_invalid_shape_2d)
{
    scnn_layer *layer = scnn_layer_alloc((scnn_layer_params){ .in_shape = { 0, 1 } });

    TEST_ASSERT_NULL(layer);

    scnn_layer_free(&layer);
}

TEST(scnn_layer, set_invalid_shape_3d)
{
    scnn_layer *layer = scnn_layer_alloc((scnn_layer_params){ .in_shape = { 1, 0, 1 } });

    TEST_ASSERT_NULL(layer);

    scnn_layer_free(&layer);
}

TEST(scnn_layer, set_invalid_shape_negative)
{
    scnn_layer *layer = scnn_layer_alloc((scnn_layer_params){ .in_shape = { -1 } });

    TEST_ASSERT_NULL(layer);

    scnn_layer_free(&layer);
}
