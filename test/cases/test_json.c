/**
 * @file test_json.c
 * @brief Unit test of json.c
 */
#include "json.h"

#include <math.h>

#include "unity.h"
#include "test_utils.h"

void setUp(void) {}

void tearDown(void) {}

#define TEST_JSON_FILE "/tmp/test.json"

void test_read_file_for_basic_types(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
        "{\n"
        "    \"number\": 3.14,\n"
        "    \"string\": \"foobar\",\n"
        "    \"bool_true\": true,\n"
        "    \"bool_false\": false,\n"
        "    \"null_value\": null\n"
        "}"
    );

    JsonObject *root_object = json_read_file(TEST_JSON_FILE);
    TEST_ASSERT_NOT_NULL(root_object);

    TEST_ASSERT_EQUAL_DOUBLE(3.14, json_get_number(root_object, "number"));

    TEST_ASSERT_EQUAL_STRING("foobar", json_get_string(root_object, "string"));

    TEST_ASSERT_TRUE(json_get_boolean(root_object, "bool_true"));
    TEST_ASSERT_FALSE(json_get_boolean(root_object, "bool_false"));

    TEST_ASSERT_EQUAL_DOUBLE(INFINITY, json_get_number(root_object, "null_value"));
    TEST_ASSERT_NULL(json_get_string(root_object, "null_value"));
    TEST_ASSERT_FALSE(json_get_boolean(root_object, "null_value"));

    json_free_object(&root_object);
    TEST_ASSERT_NULL(root_object);
}

void test_read_file_for_list(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
        "{\n"
        "    \"list\": [\n"
        "        3.14, \"foobaz\", true, null,\n"
        "        {\n"
        "            \"value\": 1\n"
        "        }\n"
        "    ]\n"
        "}"
    );

    JsonObject *root_object = json_read_file(TEST_JSON_FILE);

    JsonValue *value = json_get_value(root_object, "list");

    TEST_ASSERT_EQUAL_DOUBLE(3.14, value->number);
    value = value->next;
    TEST_ASSERT_EQUAL_STRING("foobaz", value->string);
    value = value->next;
    TEST_ASSERT_EQUAL(true, value->boolean);
    value = value->next;
    TEST_ASSERT_NULL(value->string);
    value = value->next;

    JsonObject *obj = value->object;
    JsonValue *val = json_get_value(obj, "value");
    TEST_ASSERT_EQUAL_DOUBLE(1, val->number);

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
    TEST_ASSERT_EQUAL_DOUBLE(1, json_get_number(obj1, "val1"));

    JsonObject *obj2 = json_get_child_object(root_object, "obj2");
    TEST_ASSERT_NOT_NULL(obj2);
    JsonObject *obj21 = json_get_child_object(obj2, "obj21");
    TEST_ASSERT_NOT_NULL(obj21);
    TEST_ASSERT_EQUAL_DOUBLE(2, json_get_number(obj21, "val2"));

    json_free_object(&root_object);
}
