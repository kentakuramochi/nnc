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
        "{\n"
            "\"foo\": 1,\n"
            "\"baz\": \"foobar\",\n"
            "\"pi\": 3.14,\n"
            "\"true_flag\": true,\n"
            "\"false_flag\": false\n"
        "}"
    );

    JsonObject *root_object = json_read_file(TEST_JSON_FILE);
    TEST_ASSERT_NOT_NULL(root_object);

    int integer = 0;
    json_get_integer_value(&integer, root_object, "foo");
    TEST_ASSERT_EQUAL(1, integer);

    char string[6 + 1] = { 0 };
    json_get_string_value(string, root_object, "baz");
    TEST_ASSERT_EQUAL_STRING("foobar", string);

    float flt = 0.0;
    json_get_float_value(&flt, root_object, "pi");
    TEST_ASSERT_EQUAL_FLOAT(3.14, flt);

    bool flag = false;
    json_get_boolean_value(&flag, root_object, "true_flag");
    TEST_ASSERT_EQUAL(true, flag);

    json_get_boolean_value(&flag, root_object, "false_flag");
    TEST_ASSERT_EQUAL(false, flag);

    json_free_object(&root_object);
    TEST_ASSERT_NULL(root_object);
}
