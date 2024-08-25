/**
 * @file test_nn_loss.c
 * @brief Unit tests of nn_loss.c
 *
 */
#include "nn_loss.h"

#include "unity.h"
#include "test_utils.h"

// Dummy loss type
#define NN_LOSS_TYPE_DUMMY 1

// Functions for the dummy loss
static float dummy_loss(const float *y, const float *t, const size_t size) {
    float loss = 0.0f;

    // MAE (mean absolute error)
    for (size_t i = 0; i < size; i++) {
        float d = y[i] - t[i];
        if (d < 0) {
            d *= -1;
        }
        loss += d;
    }

    loss /= size;

    return loss;
}

static void dummy_loss_backward(
    float *diff, const float *y, const float *t, const size_t size
) {
    // Backward of the MSE
    for (size_t i = 0; i < size; i++) {
        float d = y[i] - t[i];
        if (d < 0) {
            d *= -1;
        }
        diff[i] = d / size;
    }
}

float (*loss_forwards[])(const float*, const float*, const size_t) = {
    NULL,
    dummy_loss
};

void (*loss_backwards[])(float*, const float*, const float*, const size_t) = {
    NULL,
    dummy_loss_backward
};

void setUp(void) {}

void tearDown(void) {}

void test_loss(void) {
    float y[] = { 1, 0 };
    float t[] = { 0, 0 };

    TEST_ASSERT_EQUAL_FLOAT(0.5, nn_loss(NN_LOSS_TYPE_DUMMY, y, t, 2));
}

void test_loss_backward(void) {
    float y[] = { 1, 0, 0 };
    float t[] = { 0, 0, 0 };
    float dy[3];

    nn_loss_backward(dy, NN_LOSS_TYPE_DUMMY, y, t, 3);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(TEST_UTIL_FLOAT_ARRAY(0.3333333, 0, 0), dy, 3);
}
