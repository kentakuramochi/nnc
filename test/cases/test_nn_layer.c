/**
 * @file test_nn_layer.c
 * @brief Unit tests of nn_layer.c
 *
 */
#include "nn_layer.h"

#include <string.h>

#include "unity.h"

#include "activation.h"
#include "blas.h"

static NnLayerParams params = {
    .in = 3 * 28 * 28, .out = 100,
};

static NnLayer *layer;
static NnLayer *layer_next;

static float x[3 * 28 * 28];
static float dy[100];

void setUp(void) {
    layer = NULL;
    layer_next = NULL;
}

void tearDown(void) {
    nn_layer_free(&layer);
}

void test_allocate_and_free(void) {
    layer = nn_layer_alloc(params);

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

    nn_layer_free(&layer);
    TEST_ASSERT_NULL(layer);
}

void test_free_pointer_to_NULL(void) {
    nn_layer_free(&layer);
}

void test_free_NULL(void) {
    nn_layer_free(NULL);
}

void test_init(void) {
    layer = nn_layer_alloc(params);

    TEST_ASSERT_EQUAL_PTR(layer, nn_layer_init(layer));
    TEST_ASSERT_NOT_NULL(layer->x);
    TEST_ASSERT_NOT_NULL(layer->y);
    TEST_ASSERT_NOT_NULL(layer->z);
    TEST_ASSERT_NOT_NULL(layer->w);
    TEST_ASSERT_NOT_NULL(layer->b);
    TEST_ASSERT_NOT_NULL(layer->dx);
    TEST_ASSERT_NOT_NULL(layer->dz);
    TEST_ASSERT_NOT_NULL(layer->dw);
    TEST_ASSERT_NOT_NULL(layer->db);

    nn_layer_free(&layer);
}

void test_init_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_init(NULL));
}

void test_connect(void) {
    layer = nn_layer_alloc(
        (NnLayerParams){
            .in =  3 * 28 * 28, .out = 100,
        }
    );
    layer_next = nn_layer_alloc(
        (NnLayerParams){
            .out = 10,
        }
    );

    nn_layer_connect(layer, layer_next);

    TEST_ASSERT_EQUAL_INT(100, layer_next->in);

    nn_layer_free(&layer);
    nn_layer_free(&layer_next);
}

void test_forward(void) {
    layer = nn_layer_alloc(
        (NnLayerParams){
            .in=2, .out=3,
        }
    );

    nn_layer_init(layer);

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

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(z, nn_layer_forward(layer, _x), 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(answer, layer->y, 3);

    nn_layer_free(&layer);
}

void test_forward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_forward(NULL, x));
}

void test_forward_fail_if_x_is_NULL(void) {
    layer = nn_layer_alloc(params);

    TEST_ASSERT_NULL(nn_layer_forward(layer, NULL));

    nn_layer_free(&layer);
}

void test_backward(void) {
    layer = nn_layer_alloc(
        (NnLayerParams){
            .in = 2, .out = 3
        }
    );

    nn_layer_init(layer);

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

    nn_layer_forward(layer, _x);

    float _dx[] = {
        0.003130589f, 0.01056396f
    };

    float dt[] = {
        -0.01798624f, 0.99752736f, 0.99966466f
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_dx, nn_layer_backward(layer, dt), 2);

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

    nn_layer_free(&layer);
}

void test_backward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_backward(NULL, dy));
}

void test_backward_fail_if_dy_is_NULL(void) {
    layer = nn_layer_alloc(params);

    TEST_ASSERT_NULL(nn_layer_backward(layer, NULL));

    nn_layer_free(&layer);
}

void test_update(void) {
    layer = nn_layer_alloc(params);

    nn_layer_init(layer);

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

    nn_layer_update(layer, 0.01);

    float _w[] = {
        0.99, 0.98, 0.97,
        0.96, 0.95, 0.94
    };

    float _b[] = {
        0.99, 0.98, 0.97
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_w, layer->w, 2 * 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(_b, layer->b, 3);

    nn_layer_free(&layer);
}
