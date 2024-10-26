/**
 * @file test_json.c
 * @brief Unit test of json.c
 */
#include "json.h"

#include <stdio.h>

#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

#define TEST_JSON_FILE "/tmp/test.json"

// Create a test file "/tmp/test.json"
static void create_test_json(const char *string) {
    FILE *fp = fopen(TEST_JSON_FILE, "w");

    fprintf(fp, "%s\n", string);

    fclose(fp);
}

void test_read_file(void) {
    create_test_json(
        "{ \"foo\": 1 }"
    );

    JsonObject *root_object = json_read_file(TEST_JSON_FILE);
    TEST_ASSERT_NOT_NULL(root_object);

    int value = 0;
    json_get_integer_value(&value, root_object, "foo");
    TEST_ASSERT_EQUAL(1, value);

    json_free_object(&root_object);
    TEST_ASSERT_NULL(root_object);
}
