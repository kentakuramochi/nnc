/**
 * @file test_json.c
 * @brief Unit test of json.c
 */
#include "json.h"

#include <stdio.h>

#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

#define TEST_JSON_FILE "/tmp/test.json"

void test_read_file_for_basic_types(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
        "{\n"
        "    \"int_val\": 1,\n"
        "    \"float_val\": 3.14,\n"
        "    \"string\": \"foobar\",\n"
        "    \"bool_true\": true,\n"
        "    \"bool_false\": false,\n"
        "    \"null_value\": null\n"
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

void test_read_file_for_list(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
        "{\n"
        "    \"list\": [\n"
        "        3.14, \"foobaz\", true, null\n"
        "    ]\n"
        "}"
    );

    JsonObject *root_object = json_read_file(TEST_JSON_FILE);

    JsonValue *value = json_get_value(root_object, "list");

    char *list[] = { "3.14", "foobaz", "true", "null" };
    int i = 0;
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_STRING(list[i], value->string);
        value = value->next;
    }

    json_free_object(&root_object);
}

void test_read_file_for_nested_object(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
        "{\n"
        "    \"obj1\": {\n"
        "        \"val1\": 1\n"
        "    },\n"
        "    \"obj2\": {\n"
        "        \"obj21\": {\n"
        "            \"val2\": 2\n"
        "        }\n"
        "    }\n"
        "}"
    );

    JsonObject *root_object = json_read_file(TEST_JSON_FILE);

    JsonObject *obj1 = json_get_child_object(root_object, "obj1");
    TEST_ASSERT_NOT_NULL(obj1);
    int val1;
    TEST_ASSERT_TRUE(json_get_integer_value(&val1, obj1, "val1"));
    TEST_ASSERT_EQUAL_INT(1, val1);

    JsonObject *obj2 = json_get_child_object(root_object, "obj2");
    TEST_ASSERT_NOT_NULL(obj2);
    JsonObject *obj21 = json_get_child_object(obj2, "obj21");
    TEST_ASSERT_NOT_NULL(obj21);
    int val2;
    TEST_ASSERT_TRUE(json_get_integer_value(&val2, obj21, "val2"));
    TEST_ASSERT_EQUAL_INT(2, val2);

    json_free_object(&root_object);
}
