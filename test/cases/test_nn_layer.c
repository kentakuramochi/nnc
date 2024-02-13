/**
 * @file test_nn_layer.c
 * @brief Unit tests of nn_layer.c
 *
 */
#include "nn_layer.h"

#include "unity.h"
#include "test_utils.h"

#include "activation.h"
#include "blas.h"

void setUp(void) {}

void tearDown(void) {}

void test_allocate_and_free(void) {
    NnLayerParams params = {
        .batch_size = 2, .in = 3 * 28 * 28, .out = 100,
    };

    NnLayer *layer = nn_layer_alloc(params);

    TEST_ASSERT_NOT_NULL(layer);

    TEST_ASSERT_EQUAL_INT(params.batch_size, layer->batch_size);
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
    NnLayer *layer = NULL;
    nn_layer_free(&layer);
}

void test_free_NULL(void) {
    nn_layer_free(NULL);
}

void test_init(void) {
    NnLayerParams params = {
        .batch_size = 1, .in = 2, .out = 3
    };

    NnLayer *layer = nn_layer_alloc(params);

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
    NnLayer layer = {
        .batch_size = 2, .in = 2, .out = 10,
    };

    NnLayer next_layer = {
        .out = 1,
    };

    nn_layer_connect(&layer, &next_layer);

    TEST_ASSERT_EQUAL_INT(2, next_layer.batch_size);
    TEST_ASSERT_EQUAL_INT(10, next_layer.in);
}

void test_forward(void) {
    NnLayer layer = {
        .batch_size = 1,
        .in = 2,
        .out = 3,
        .x = FLOAT_ZEROS(2),
        .y = FLOAT_ZEROS(3),
        .z = FLOAT_ZEROS(3),
        .w = FLOAT_ZEROS(3 * 2),
        .b = FLOAT_ZEROS(3),
        .dx = FLOAT_ZEROS(2),
        .dz = FLOAT_ZEROS(3),
        .dw = FLOAT_ZEROS(3 * 2),
        .db = FLOAT_ZEROS(3)
    };

    COPY_ARRAY(
        layer.w,
        FLOAT_ARRAY(
            0, 1,
            2, 3,
            4, 5
        )
    );

    COPY_ARRAY(
        layer.b,
        FLOAT_ARRAY(1, 1, 1)
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(0.982014f, 0.997527f, 0.999665f),
        nn_layer_forward(&layer, FLOAT_ARRAY(1, 1)),
        3
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(4, 6, 8),
        layer.y,
        3
    );
}

void test_forward_batch(void) {
    NnLayer layer = {
        .batch_size = 2,
        .in = 2,
        .out = 3,
        .x = FLOAT_ZEROS(2 * 2),
        .y = FLOAT_ZEROS(2 * 3),
        .z = FLOAT_ZEROS(2 * 3),
        .w = FLOAT_ZEROS(3 * 2),
        .b = FLOAT_ZEROS(3),
        .dx = FLOAT_ZEROS(2 * 2),
        .dz = FLOAT_ZEROS(2 * 3),
        .dw = FLOAT_ZEROS(3 * 2),
        .db = FLOAT_ZEROS(3)
    };

    COPY_ARRAY(
        layer.w,
        FLOAT_ARRAY(
            0, 1,
            2, 3,
            4, 5
        )
    );

    COPY_ARRAY(
        layer.b,
        FLOAT_ARRAY(1, 1, 1)
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(
            0.982014f, 0.997527f, 0.999665f,
            0.982014f, 0.997527f, 0.999665f
        ),
        nn_layer_forward(
            &layer,
            FLOAT_ARRAY(
                1, 1,
                1, 1
            )
        ),
        (2 * 3)
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(
            4, 6, 8,
            4, 6, 8
        ),
        layer.y,
        (2 * 3)
    );
}

void test_forward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_forward(NULL, FLOAT_ZEROS(1)));
}

void test_forward_fail_if_x_is_NULL(void) {
    NnLayer layer = {
        .batch_size = 1,
        .in = 2,
        .out = 3,
        .x = FLOAT_ZEROS(2),
        .y = FLOAT_ZEROS(3),
        .z = FLOAT_ZEROS(3),
        .w = FLOAT_ZEROS(3 * 2),
        .b = FLOAT_ZEROS(3),
        .dx = FLOAT_ZEROS(2),
        .dz = FLOAT_ZEROS(3),
        .dw = FLOAT_ZEROS(3 * 2),
        .db = FLOAT_ZEROS(3)
    };

    TEST_ASSERT_NULL(nn_layer_forward(&layer, NULL));
}

void test_backward(void) {
    NnLayer layer = {
        .batch_size = 1,
        .in = 2,
        .out = 3,
        .x = FLOAT_ZEROS(2),
        .y = FLOAT_ZEROS(3),
        .z = FLOAT_ZEROS(3),
        .w = FLOAT_ZEROS(3 * 2),
        .b = FLOAT_ZEROS(3),
        .dx = FLOAT_ZEROS(2),
        .dz = FLOAT_ZEROS(3),
        .dw = FLOAT_ZEROS(3 * 2),
        .db = FLOAT_ZEROS(3)
    };

    COPY_ARRAY(
        layer.w,
        FLOAT_ARRAY(
            0, 1,
            2, 3,
            4, 5
        )
    );

    COPY_ARRAY(
        layer.b,
        FLOAT_ARRAY(1, 1, 1)
    );

    nn_layer_forward(&layer, FLOAT_ARRAY(1, 1));

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(0.003130589f, 0.01056396f),
        nn_layer_backward(
            &layer,
            FLOAT_ARRAY(-0.01798624f, 0.99752736f, 0.99966466f)
        ),
        2
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(
            -0.0003176862f, 0.002460367f, 0.0003351109f,
            -0.0003176862f, 0.002460367f, 0.0003351109f
        ),
        layer.dw,
        (3 * 2)
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(-0.0003176862f, 0.002460367f, 0.0003351109f),
        layer.db,
        3
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(-0.0003176862f, 0.002460367f, 0.0003351109f),
        layer.dz,
        3
    );
}

void test_backward_fail_if_layer_is_NULL(void) {
    TEST_ASSERT_NULL(nn_layer_backward(NULL, FLOAT_ZEROS(1)));
}

void test_backward_fail_if_dy_is_NULL(void) {
    NnLayer layer = {
        .batch_size = 1,
        .in = 2,
        .out = 3,
        .x = FLOAT_ZEROS(2),
        .y = FLOAT_ZEROS(3),
        .z = FLOAT_ZEROS(3),
        .w = FLOAT_ZEROS(3 * 2),
        .b = FLOAT_ZEROS(3),
        .dx = FLOAT_ZEROS(2),
        .dz = FLOAT_ZEROS(3),
        .dw = FLOAT_ZEROS(3 * 2),
        .db = FLOAT_ZEROS(3)
    };

    TEST_ASSERT_NULL(nn_layer_backward(&layer, NULL));
}

void test_update(void) {
    NnLayer layer = {
        .in = 2,
        .out = 3,
        .x = FLOAT_ZEROS(2),
        .y = FLOAT_ZEROS(3),
        .z = FLOAT_ZEROS(3),
        .w = FLOAT_ZEROS(3 * 2),
        .b = FLOAT_ZEROS(3),
        .dx = FLOAT_ZEROS(2),
        .dz = FLOAT_ZEROS(3),
        .dw = FLOAT_ZEROS(3 * 2),
        .db = FLOAT_ZEROS(3)
    };

    COPY_ARRAY(
        layer.w,
        FLOAT_ARRAY(
            1, 1,
            1, 1,
            1, 1
        )
    );

    COPY_ARRAY(
        layer.dw,
        FLOAT_ARRAY(
            1, 2,
            3, 4,
            5, 6
        )
    );

    COPY_ARRAY(
        layer.b,
        FLOAT_ARRAY(1, 1, 1)
    );

    COPY_ARRAY(
        layer.db,
        FLOAT_ARRAY(1, 2, 3)
    );

    nn_layer_update(&layer, 0.01);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(
            0.99, 0.98, 0.97,
            0.96, 0.95, 0.94
        ),
        layer.w,
        (3 * 2)
    );

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(
        FLOAT_ARRAY(0.99, 0.98, 0.97),
        layer.b,
        3
    );
}
