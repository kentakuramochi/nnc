/**
 * @file test_trainer.c
 * @brief Unit tests of trainer.c
 */
#include "util.h"

#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

void test_alloc_and_free_dataset(void) {
    const int data_num = 100;
    const int vec_size = 3;

    float **dataset = alloc_dataset(data_num, vec_size);

    TEST_ASSERT_NOT_NULL(dataset);
    for (int i = 0; i < data_num; i++) {
        TEST_ASSERT_NOT_NULL(dataset[i]);
    }

    free_dataset(&dataset, data_num);
    TEST_ASSERT_NULL(dataset);
}

void test_allocation_fail_if_size_is_zero(void) {
    TEST_ASSERT_NULL(alloc_dataset(0, 1));
    TEST_ASSERT_NULL(alloc_dataset(1, 0));
}

void test_free_null_dataset(void) {
    free_dataset(NULL, 1);

    float **dataset = NULL;
    free_dataset(&dataset, 1);
}

void test_argmax(void) {
    float vector[] = { 1, 3, -1 };

    TEST_ASSERT_EQUAL(1, argmax(vector, 3));
}

void test_fail_argmax(void) {
    TEST_ASSERT_EQUAL(-1, argmax(NULL, 3));

    float vector[] = { 1, 3, -1 };

    TEST_ASSERT_EQUAL(-1, argmax(vector, 0));
}