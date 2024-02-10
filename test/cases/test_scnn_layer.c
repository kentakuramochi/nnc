/**
 * @file test_scnn_layer.c
 * @brief Unit tests of scnn_layer.c
 * 
 */
#include "scnn_layer.h"

#include <string.h>

#include "unity.h"

#include "activation.h"
#include "scnn_blas.h"

static scnn_layer_params params = {
    .in = 3 * 28 * 28, .out = 100,
};

static scnn_layer *layer;
static scnn_layer *layer_next;

static float x[3 * 28 * 28];
static float dy[100];

void setUp(void)
{
    layer = NULL;
    layer_next = NULL;
}

void tearDown(void)
{
    scnn_layer_free(&layer);
}

void test_allocate_and_free(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL_INT(params.in, layer->in);
    TEST_ASSERT_EQUAL(params.out, layer->out);

    TEST_ASSERT_NULL(layer->x);
    TEST_ASSERT_NULL(layer->y);
    TEST_ASSERT_NULL(layer->z);
    TEST_ASSERT_NULL(layer->w);
    TEST_ASSERT_NULL(layer->b);
    TEST_ASSERT_NULL(layer->dx);
    TEST_ASSERT_NULL(layer->dz);
    TEST_ASSERT_NULL(layer->dw);
    TEST_ASSERT_NULL(layer->db);

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

void test_init(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_EQUAL_PTR(layer, scnn_layer_init(layer));
    TEST_ASSERT_NOT_NULL(layer->x);
    TEST_ASSERT_NOT_NULL(layer->y);
    TEST_ASSERT_NOT_NULL(layer->z);
    TEST_ASSERT_NOT_NULL(layer->w);
    TEST_ASSERT_NOT_NULL(layer->b);
    TEST_ASSERT_NOT_NULL(layer->dx);
    TEST_ASSERT_NOT_NULL(layer->dz);
    TEST_ASSERT_NOT_NULL(layer->dw);
    TEST_ASSERT_NOT_NULL(layer->db);

    scnn_layer_free(&layer);
}

void test_init_fail_if_layer_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_layer_init(NULL));
}

void test_connect(void)
{
    layer = scnn_layer_alloc(
        (scnn_layer_params){
            .in =  3 * 28 * 28, .out = 100,
        }
    );
    layer_next = scnn_layer_alloc(
        (scnn_layer_params){
            .out = 10,
        }
    );

    scnn_layer_connect(layer, layer_next);

    TEST_ASSERT_EQUAL_INT(100, layer_next->in);

    scnn_layer_free(&layer);
    scnn_layer_free(&layer_next);
}

void test_forward(void)
{
    layer = scnn_layer_alloc(
        (scnn_layer_params){
            .in=2, .out=3,
        }
    );

    scnn_layer_init(layer);

    float w[] = {
        0, 1, 2,
        3, 4, 5
    };
    memcpy(layer->w, w, sizeof(w));

    float b[] = {
        1, 1, 1
    };
    memcpy(layer->b, b, sizeof(b));

    float _x[] = {
        1, 1
    };

    float answer[] = {
        4, 6, 8
    };

    float z[] = {
        0.982014f, 0.997527f, 0.999665f
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(z, scnn_layer_forward(layer, _x), 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, layer->y, 3);

    scnn_layer_free(&layer);
}

void test_forward_fail_if_layer_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_layer_forward(NULL, x));
}

void test_forward_fail_if_x_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NULL(scnn_layer_forward(layer, NULL));

    scnn_layer_free(&layer);
}

void test_backward(void)
{
    layer = scnn_layer_alloc(
        (scnn_layer_params){
            .in = 2, .out = 3
        }
    );

    scnn_layer_init(layer);

    float w[] = {
        0, 1, 2,
        3, 4, 5
    };
    memcpy(layer->w, w, sizeof(w));

    float b[] = {
        1, 1, 1
    };
    memcpy(layer->b, b, sizeof(b));

    float _x[] = {
        1, 1
    };

    scnn_layer_forward(layer, _x);

    float _dx[] = {
        0.003130589f, 0.01056396f
    };

    float dt[] = {
        -0.01798624f, 0.99752736f, 0.99966466f
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_dx, scnn_layer_backward(layer, dt), 2);

    float _dw[] = {
        -0.0003176862f, 0.002460367f, 0.0003351109f,
        -0.0003176862f, 0.002460367f, 0.0003351109f
    };

    float _db[] = {
        -0.0003176862f, 0.002460367f, 0.0003351109f
    };

    float dz[] = {
        -0.0003176862f, 0.002460367f, 0.0003351109f
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_dw, layer->dw, 2 * 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_db, layer->db, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dz, layer->dz, 3);

    scnn_layer_free(&layer);
}

void test_backward_fail_if_layer_is_NULL(void)
{
    TEST_ASSERT_NULL(scnn_layer_backward(NULL, dy));
}

void test_backward_fail_if_dy_is_NULL(void)
{
    layer = scnn_layer_alloc(params);

    TEST_ASSERT_NULL(scnn_layer_backward(layer, NULL));

    scnn_layer_free(&layer);
}

void test_update(void)
{
    layer = scnn_layer_alloc(params);

    scnn_layer_init(layer);

    float w[] = {
        1, 1, 1,
        1, 1, 1
    };
    memcpy(layer->w, w, sizeof(w));

    float dw[] = {
        1, 2, 3,
        4, 5, 6
    };
    memcpy(layer->dw, dw, sizeof(dw));

    float b[] = {
        1, 1, 1
    };
    memcpy(layer->b, b, sizeof(b));

    float db[] = {
        1, 2, 3
    };
    memcpy(layer->db, db, sizeof(db));

    layer_update(layer, 0.01);

    float _w[] = {
        0.99, 0.98, 0.97,
        0.96, 0.95, 0.94
    };

    float _b[] = {
        0.99, 0.98, 0.97
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_w, layer->w, 2 * 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_b, layer->b, 3);

    scnn_layer_free(&layer);
}
