/**
 * @file test_scnn_layer.c
 * @brief Unit tests of scnn_layer.c
 * 
 */
#include "scnn_layer.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_layer);

scnn_layer *layer;

TEST_SETUP(scnn_layer)
{
    layer = NULL;
}

TEST_TEAR_DOWN(scnn_layer)
{
    scnn_layer_free(&layer);

    TEST_ASSERT_NULL(layer);
}

TEST(scnn_layer, allocate_layer)
{
    scnn_layer_params params = { .in_shape = { 1, 3, 28, 28 } };
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL(SCNN_LAYER_NONE, layer->params.type);
    TEST_ASSERT_EQUAL_INT_ARRAY(params.in_shape, layer->params.in_shape, 4);

    TEST_ASSERT_EQUAL(0, layer->params.id);
    TEST_ASSERT_EQUAL(0, layer->params.prev_id);
    TEST_ASSERT_EQUAL(0, layer->params.next_id);

    TEST_ASSERT_NULL(layer->x);
    TEST_ASSERT_NULL(layer->y);
    TEST_ASSERT_NULL(layer->w);
    TEST_ASSERT_NULL(layer->b);
    TEST_ASSERT_NULL(layer->dx);
    TEST_ASSERT_NULL(layer->dw);
    TEST_ASSERT_NULL(layer->db);

    TEST_ASSERT_EQUAL(NULL, layer->init);

    TEST_ASSERT_EQUAL(NULL, layer->forward);
    TEST_ASSERT_EQUAL(NULL, layer->backward);
}

TEST(scnn_layer, free_NULL)
{
    scnn_layer_free(NULL);
}

TEST(scnn_layer, free_twice)
{
    scnn_layer_params params = { .in_shape = { 1, 3, 28, 28 } };
    scnn_layer *layer = scnn_layer_alloc(params);

    scnn_layer_free(&layer);

    // 2nd freeing is done in TEST_TEAR_DOWN
}
