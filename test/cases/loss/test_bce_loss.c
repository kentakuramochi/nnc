/**
 * @file test_bce_loss.c
 * @brief Unit tests of bce_loss.c
 *
 */
#include "bce_loss.h"

#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void test_bce_loss(void) {
    float y[] = { 0.7, 0.3 };
    float t[] = { 1, 0 };

    TEST_ASSERT_EQUAL_FLOAT(0.514573, bce_loss(y, t, 2));
}
