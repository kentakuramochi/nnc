/**
 * @file test_scnn_layer.c
 * @brief Unit tests of scnn_layer.c
 * 
 */
#include "scnn_layer.h"

// Private header, include to verify private members
#include "impl/scnn_layer_impl.h"

#include "unity.h"

#include "mock_scnn_mat.h"

static scnn_layer *layer;
static scnn_layer *layer2;

static scnn_layer_params params = {
    .in_shape = { 1, 3, 28, 28 }
};

void setUp(void)
{
    layer = NULL;
    layer2 = NULL;
}

void tearDown(void)
{}

void test_allocate_and_free(void)
{
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

    scnn_mat_free_Expect(&layer->x);
    scnn_mat_free_Expect(&layer->y);
    scnn_mat_free_Expect(&layer->w);
    scnn_mat_free_Expect(&layer->b);
    scnn_mat_free_Expect(&layer->dx);
    scnn_mat_free_Expect(&layer->dw);
    scnn_mat_free_Expect(&layer->db);
    scnn_layer_free(&layer);
    TEST_ASSERT_NULL(layer);
}

void test_free_pointer_to_NULL(void)
{
    scnn_layer_free(&layer);
}

void test_free_NULL(void)
{
    scnn_layer_free(NULL);
}

static scnn_layer *dummy_init(scnn_layer *layer)
{
    return layer;
}

void test_init(void)
{
    layer = scnn_layer_alloc(params);

    layer->init = dummy_init;

    TEST_ASSERT_EQUAL_PTR(layer, scnn_layer_init(layer));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_init_fail_if_layer_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    layer->init = dummy_init;

    TEST_ASSERT_NULL(scnn_layer_init(NULL));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_init_fail_if_init_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NULL(scnn_layer_init(layer));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_connect(void)
{
    layer = scnn_layer_alloc(params);
    layer->params.id = 1;
    layer2 = scnn_layer_alloc(params);
    layer2->params.id = 2;

    scnn_layer_connect(layer, layer2);

    TEST_ASSERT_EQUAL_INT(1, layer->params.id);
    TEST_ASSERT_EQUAL_INT(0, layer->params.prev_id);
    TEST_ASSERT_EQUAL_INT(2, layer->params.next_id);

    TEST_ASSERT_EQUAL_INT(2, layer2->params.id);
    TEST_ASSERT_EQUAL_INT(1, layer2->params.prev_id);
    TEST_ASSERT_EQUAL_INT(0, layer2->params.next_id);

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
    scnn_layer_free(&layer2);
}

static scnn_dtype dummy_y;
static scnn_dtype *dummy_forward_plus1(scnn_layer *layer, const scnn_dtype *x)
{
    dummy_y = 1 + *x;
    return &dummy_y;
}

void test_forward(void)
{
    layer = scnn_layer_alloc(params);

    layer->forward = dummy_forward_plus1;

    scnn_dtype x = 1;
    TEST_ASSERT_EQUAL_PTR(&dummy_y, scnn_layer_forward(layer, &x));
    TEST_ASSERT_EQUAL_FLOAT(2, dummy_y);

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_forward_fail_if_layer_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    layer->forward = dummy_forward_plus1;

    scnn_dtype x = 1;
    TEST_ASSERT_NULL(scnn_layer_forward(NULL, &x));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_forward_fail_if_x_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    layer->forward = dummy_forward_plus1;

    TEST_ASSERT_NULL(scnn_layer_forward(layer, NULL));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_forward_fail_if_forward_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    scnn_dtype x = 1;
    TEST_ASSERT_NULL(scnn_layer_forward(layer, &x));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

static scnn_dtype dummy_dx;
static scnn_dtype *dummy_backward_minus2(scnn_layer *layer, const scnn_dtype *dy)
{
    dummy_dx = *dy - 2;
    return &dummy_dx;
}

void test_backward(void)
{
    layer = scnn_layer_alloc(params);

    layer->backward = dummy_backward_minus2;

    scnn_dtype dy = 1;
    TEST_ASSERT_EQUAL_PTR(&dummy_dx, scnn_layer_backward(layer, &dy));
    TEST_ASSERT_EQUAL_FLOAT(-1, dummy_dx);

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_backward_fail_if_layer_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    layer->backward = dummy_backward_minus2;

    scnn_dtype dy = 1;
    TEST_ASSERT_NULL(scnn_layer_backward(NULL, &dy));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_backward_fail_if_dy_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    layer->backward = dummy_backward_minus2;

    TEST_ASSERT_NULL(scnn_layer_backward(layer, NULL));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}

void test_backward_fail_if_backward_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    scnn_dtype dy = 1;
    TEST_ASSERT_NULL(scnn_layer_backward(layer, &dy));

    scnn_mat_free_Ignore();
    scnn_layer_free(&layer);
}
