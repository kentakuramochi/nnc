/**
 * @file test_scnn_fc.c
 * @brief Unit tests of scnn_fc.c
 * 
 */
#include "scnn_fc.h"

#include "scnn_blas.h"

#include "unity_fixture.h"

TEST_GROUP(scnn_fc);

scnn_layer *fc;

TEST_SETUP(scnn_fc)
{
    fc = NULL;
}

TEST_TEAR_DOWN(scnn_fc)
{
    scnn_layer_free(&fc);

    TEST_ASSERT_NULL(fc);
}

TEST(scnn_fc, allocate_fc_layer)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 28, 28 }, .out=10 };
    fc = scnn_fc_layer(params);

    TEST_ASSERT_NOT_NULL(fc);

    TEST_ASSERT_EQUAL(SCNN_LAYER_FC, fc->params.type);
    TEST_ASSERT_EQUAL_INT(1, fc->params.in_shape[0]);
    TEST_ASSERT_EQUAL_INT(2, fc->params.in_shape[1]);
    TEST_ASSERT_EQUAL_INT(28, fc->params.in_shape[2]);
    TEST_ASSERT_EQUAL_INT(28, fc->params.in_shape[3]);
    TEST_ASSERT_EQUAL_INT(10, fc->params.out);

    TEST_ASSERT_NOT_NULL(fc->init);

    TEST_ASSERT_NOT_NULL(fc->forward);
    TEST_ASSERT_NOT_NULL(fc->backward);
}

TEST(scnn_fc, initialize)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 28, 28 }, .out=10 };
    fc = scnn_fc_layer(params);

    TEST_ASSERT_NOT_NULL(fc->init(fc));

    TEST_ASSERT_NOT_NULL(fc->x);
    TEST_ASSERT_NOT_NULL(fc->x->data);
    TEST_ASSERT_EQUAL_INT(1, fc->x->shape[0]);
    TEST_ASSERT_EQUAL_INT(2, fc->x->shape[1]);
    TEST_ASSERT_EQUAL_INT(28, fc->x->shape[2]);
    TEST_ASSERT_EQUAL_INT(28, fc->x->shape[3]);
    TEST_ASSERT_EQUAL_INT((2 * 28 * 28), fc->x->size);

    TEST_ASSERT_NOT_NULL(fc->y);
    TEST_ASSERT_NOT_NULL(fc->y->data);
    TEST_ASSERT_EQUAL_INT(1, fc->y->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, fc->y->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, fc->y->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, fc->y->shape[3]);
    TEST_ASSERT_EQUAL_INT(10, fc->y->size);

    TEST_ASSERT_NOT_NULL(fc->w);
    TEST_ASSERT_NOT_NULL(fc->w->data);
    TEST_ASSERT_EQUAL_INT((2 * 28 * 28), fc->w->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, fc->w->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, fc->w->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, fc->w->shape[3]);
    TEST_ASSERT_EQUAL_INT((10 * 2 * 28 * 28), fc->w->size);

    TEST_ASSERT_NOT_NULL(fc->b);
    TEST_ASSERT_NOT_NULL(fc->b->data);
    TEST_ASSERT_EQUAL_INT(1, fc->b->shape[0]);
    TEST_ASSERT_EQUAL_INT(10, fc->b->shape[1]);
    TEST_ASSERT_EQUAL_INT(1, fc->b->shape[2]);
    TEST_ASSERT_EQUAL_INT(1, fc->b->shape[3]);
    TEST_ASSERT_EQUAL_INT(10, fc->b->size);

    TEST_ASSERT_NOT_NULL(fc->dx);
    TEST_ASSERT_NOT_NULL(fc->dx->data);
    TEST_ASSERT_EQUAL_INT(fc->x->shape[0], fc->dx->shape[0]);
    TEST_ASSERT_EQUAL_INT(fc->x->shape[1], fc->dx->shape[1]);
    TEST_ASSERT_EQUAL_INT(fc->x->shape[2], fc->dx->shape[2]);
    TEST_ASSERT_EQUAL_INT(fc->x->shape[3], fc->dx->shape[3]);
    TEST_ASSERT_EQUAL_INT(fc->x->size, fc->dx->size);

    TEST_ASSERT_NOT_NULL(fc->dw);
    TEST_ASSERT_NOT_NULL(fc->dw->data);
    TEST_ASSERT_EQUAL_INT(fc->w->shape[0], fc->dw->shape[0]);
    TEST_ASSERT_EQUAL_INT(fc->w->shape[1], fc->dw->shape[1]);
    TEST_ASSERT_EQUAL_INT(fc->w->shape[2], fc->dw->shape[2]);
    TEST_ASSERT_EQUAL_INT(fc->w->shape[3], fc->dw->shape[3]);
    TEST_ASSERT_EQUAL_INT(fc->w->size, fc->dw->size);

    TEST_ASSERT_NOT_NULL(fc->db);
    TEST_ASSERT_NOT_NULL(fc->db->data);
    TEST_ASSERT_EQUAL_INT(fc->b->shape[0], fc->db->shape[0]);
    TEST_ASSERT_EQUAL_INT(fc->b->shape[1], fc->db->shape[1]);
    TEST_ASSERT_EQUAL_INT(fc->b->shape[2], fc->db->shape[2]);
    TEST_ASSERT_EQUAL_INT(fc->b->shape[3], fc->db->shape[3]);
    TEST_ASSERT_EQUAL_INT(fc->b->size, fc->db->size);
}

TEST(scnn_fc, cannot_initialize_with_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 28, 28 }, .out=10 };
    fc = scnn_fc_layer(params);

    TEST_ASSERT_NULL(fc->init(NULL));

    TEST_ASSERT_NULL(fc->x);
    TEST_ASSERT_NULL(fc->y);
    TEST_ASSERT_NULL(fc->w);
    TEST_ASSERT_NULL(fc->b);
    TEST_ASSERT_NULL(fc->dx);
    TEST_ASSERT_NULL(fc->dw);
    TEST_ASSERT_NULL(fc->db);
}

TEST(scnn_fc, cannot_initialize_with_invalid_in_shape)
{
    scnn_layer_params params = { .in_shape={ -1, 2, 28, 28 }, .out=10 };
    fc = scnn_fc_layer(params);

    TEST_ASSERT_NULL(fc->init(fc));

    TEST_ASSERT_NULL(fc->x);
    TEST_ASSERT_NULL(fc->y);
    TEST_ASSERT_NULL(fc->w);
    TEST_ASSERT_NULL(fc->b);
    TEST_ASSERT_NULL(fc->dx);
    TEST_ASSERT_NULL(fc->dw);
    TEST_ASSERT_NULL(fc->db);
}

TEST(scnn_fc, cannot_initialize_with_invalid_out)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 28, 28 }, .out=0 };
    fc = scnn_fc_layer(params);

    TEST_ASSERT_NULL(fc->init(fc));

    TEST_ASSERT_NULL(fc->x);
    TEST_ASSERT_NULL(fc->y);
    TEST_ASSERT_NULL(fc->w);
    TEST_ASSERT_NULL(fc->b);
    TEST_ASSERT_NULL(fc->dx);
    TEST_ASSERT_NULL(fc->dw);
    TEST_ASSERT_NULL(fc->db);
}

TEST(scnn_fc, forward)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype x[] = {
        1, 1
    };

    fc->forward(fc, x);

    scnn_dtype y[] = {
        4, 6, 8
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, fc->y->data, 3);
}

TEST(scnn_fc, forward_with_batch_dim)
{
    scnn_layer_params params = { .in_shape={ 3, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype x[] = {
        1, 1,
        1, 2,
        2, 2
    };

    fc->forward(fc, x);

    scnn_dtype y[] = {
        4, 6, 8,
        7, 10, 13,
        7, 11, 15
    };

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, fc->y->data, (3 * 3));
}

TEST(scnn_fc, forward_fails_when_x_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype y[] = {
        0, 1, 2
    };
    scnn_scopy(3, y, 1, fc->y->data, 1);

    fc->forward(fc, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, fc->y->data, 3);
}

TEST(scnn_fc, forward_fails_when_layer_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype y[] = {
        0, 1, 2
    };
    scnn_scopy(3, y, 1, fc->y->data, 1);

    scnn_dtype x[] = {
        1, 1
    };

    fc->forward(NULL, x);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(y, fc->y->data, 3);
}

TEST(scnn_fc, backward)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype x[] = {
        1, 2
    };
    fc->forward(fc, x);

    scnn_dtype dy[] = {
        8, 12, 16
    };
    fc->backward(fc, dy);

    scnn_dtype dx[] = {
        44, 152
    };
    scnn_dtype dw[] = {
        8,  12, 16,
        16, 24, 32
    };
    scnn_dtype db[] = {
        8, 12, 16
    };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, fc->dx->data, fc->dx->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dw, fc->dw->data, fc->dw->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(db, fc->db->data, fc->db->size);
}

TEST(scnn_fc, backward_with_batch_dim)
{
    scnn_layer_params params = { .in_shape={ 3, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype x[] = {
        1, 1,
        1, 2,
        2, 2
    };
    fc->forward(fc, x);

    scnn_dtype dy[] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    };
    fc->backward(fc, dy);

    scnn_dtype dx[] = {
        5, 14,
        14, 50,
        23, 86
    };
    scnn_dtype dw[] = {
        15, 19, 23,
        18, 23, 28
    };
    scnn_dtype db[] = {
        3, 4, 5
    };
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, fc->dx->data, fc->dx->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dw, fc->dw->data, fc->dw->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(db, fc->db->data, fc->db->size);
}

TEST(scnn_fc, backward_fails_when_dy_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype dx[] = {
        0, 1
    };
    scnn_scopy(2, dx, 1, fc->dx->data, 1);

    scnn_dtype dw[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), dw, 1, fc->dw->data, 1);

    scnn_dtype db[] = {
        0, 1, 2
    };
    scnn_scopy(3, db, 1, fc->db->data, 1);

    scnn_dtype x[] = {
        1, 2
    };
    fc->forward(fc, x);

    fc->backward(fc, NULL);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, fc->dx->data, fc->dx->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dw, fc->dw->data, fc->dw->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(db, fc->db->data, fc->db->size);
}

TEST(scnn_fc, backward_fails_when_layer_is_NULL)
{
    scnn_layer_params params = { .in_shape={ 1, 2, 1, 1 }, .out=3 };
    fc = scnn_fc_layer(params);
    fc->init(fc);

    scnn_dtype w[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), w, 1, fc->w->data, 1);

    scnn_dtype b[] = {
        1, 1, 1
    };
    scnn_scopy(3, b, 1, fc->b->data, 1);

    scnn_dtype dx[] = {
        0, 1
    };
    scnn_scopy(2, dx, 1, fc->dx->data, 1);

    scnn_dtype dw[] = {
        0, 1, 2,
        3, 4, 5
    };
    scnn_scopy((2 * 3), dw, 1, fc->dw->data, 1);

    scnn_dtype db[] = {
        0, 1, 2
    };
    scnn_scopy(3, db, 1, fc->db->data, 1);

    scnn_dtype x[] = {
        1, 2
    };
    fc->forward(fc, x);

    scnn_dtype dy[] = {
        8, 12, 16
    };
    fc->backward(NULL, dy);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dx, fc->dx->data, fc->dx->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(dw, fc->dw->data, fc->dw->size);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(db, fc->db->data, fc->db->size);
}
