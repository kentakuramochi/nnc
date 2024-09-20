/**
 * @file test_trainer.c
 * @brief Unit tests of trainer.c
 */
#include "util.h"

#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void test_alloc_dataset(void) {
    float **dataset = alloc_dataset(100, 10);

    TEST_ASSERT_NOT_NULL(dataset);

    free_dataset(dataset, 100);
}
