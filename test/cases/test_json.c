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

void test_read_file_for_basic_types(void) {
    create_test_json(
        "{\n"
            "\"int_val\": 1,\n"
            "\"float_val\": 3.14,\n"
            "\"string\": \"foobar\",\n"
            "\"bool_true\": true,\n"
            "\"bool_false\": false,\n"
            "\"null_value\": null\n"
        "}"
    );

    JsonObject *root_object = json_read_file(TEST_JSON_FILE);
    TEST_ASSERT_NOT_NULL(root_object);

    int int_val = 0;
    TEST_ASSERT_TRUE(json_get_integer_value(&int_val, root_object, "int_val"));
    TEST_ASSERT_EQUAL(1, int_val);

    float float_val = 0.0;
    TEST_ASSERT_TRUE(json_get_float_value(&float_val, root_object, "float_val"));
    TEST_ASSERT_EQUAL_FLOAT(3.14, float_val);

    char string[6 + 1] = { 'x','x','x','x','x','x','x' };
    TEST_ASSERT_TRUE(json_get_string_value(string, root_object, "string"));
    TEST_ASSERT_EQUAL_STRING("foobar", string);

    bool flag = false;
    TEST_ASSERT_TRUE(json_get_boolean_value(&flag, root_object, "bool_true"));
    TEST_ASSERT_EQUAL(true, flag);
    TEST_ASSERT_TRUE(json_get_boolean_value(&flag, root_object, "bool_false"));
    TEST_ASSERT_EQUAL(false, flag);

    TEST_ASSERT_FALSE(json_get_integer_value(&int_val, root_object, "null_value"));
    TEST_ASSERT_FALSE(json_get_float_value(&float_val, root_object, "null_value"));
    TEST_ASSERT_FALSE(json_get_string_value(string, root_object, "null_value"));
    TEST_ASSERT_FALSE(json_get_boolean_value(&flag, root_object, "null_value"));

    json_free_object(&root_object);
    TEST_ASSERT_NULL(root_object);
}
