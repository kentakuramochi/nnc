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
    scnn_layer *layer = scnn_layer_alloc();

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL(0, layer->id);

    TEST_ASSERT_EQUAL(0, layer->prev_id);
    TEST_ASSERT_EQUAL(0, layer->next_id);

    TEST_ASSERT_EQUAL(NULL, layer->forward);
    TEST_ASSERT_EQUAL(NULL, layer->backward);

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
