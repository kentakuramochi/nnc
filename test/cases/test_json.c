/**
 * @file test_json.c
 * @brief Unit test of json.c
 */
#include "json.h"

#include <math.h>

#include "unity.h"
#include "test_utils.h"

#define TEST_JSON_FILE "/tmp/test.json"

// Create a JSON file for the test
void create_test_json(void) {
    test_util_create_text_file(
        TEST_JSON_FILE,
            "{"
            "    \"pi\": 3.14,\n"
            "    \"name\": \"John Doe\",\n"
            "    \"is_tested\": true,\n"
            "    \"is_ignored\": false,\n"
            "    \"dangling\": null,\n"
            "    \"list\": [\n"
            "        123, \"foobar\", true, null,\n"
            "        {\n"
            "            \"index\": 5\n"
            "        }\n"
            "    ],\n"
            "    \"object\": {\n"
            "        \"value\": 42\n"
            "    }\n"
            "}"
    );
}

JsonObject *object;

void setUp(void) {
    create_test_json();

    object = json_read_file(TEST_JSON_FILE);
}

void tearDown(void) {
    json_free_object(&object);
}

void test_read_file_and_free(void) {
    TEST_ASSERT_NOT_NULL(object);

    json_free_object(&object);
    TEST_ASSERT_NULL(object);
}

void test_get_number(void) {
    TEST_ASSERT_EQUAL_DOUBLE(3.14, json_get_number(object, "pi"));

    TEST_ASSERT_EQUAL_DOUBLE(INFINITY, json_get_number(object, "name"));
    TEST_ASSERT_EQUAL_DOUBLE(INFINITY, json_get_number(object, "is_tested"));
    TEST_ASSERT_EQUAL_DOUBLE(INFINITY, json_get_number(object, "is_ignored"));
    TEST_ASSERT_EQUAL_DOUBLE(INFINITY, json_get_number(object, "dangling"));

    TEST_ASSERT_EQUAL_DOUBLE(INFINITY, json_get_number(object, "undefined"));
}

void test_get_string(void) {
    TEST_ASSERT_EQUAL_STRING("John Doe", json_get_string(object, "name"));

    TEST_ASSERT_NULL(json_get_string(object, "pi"));
    TEST_ASSERT_NULL(json_get_string(object, "is_tested"));
    TEST_ASSERT_NULL(json_get_string(object, "is_ignored"));
    TEST_ASSERT_NULL(json_get_string(object, "dangling"));

    TEST_ASSERT_NULL(json_get_string(object, "undefined"));
}

void test_get_boolean(void) {
    TEST_ASSERT_TRUE(json_get_boolean(object, "is_tested"));
    TEST_ASSERT_FALSE(json_get_boolean(object, "is_ignored"));

    TEST_ASSERT_FALSE(json_get_boolean(object, "pi"));
    TEST_ASSERT_FALSE(json_get_boolean(object, "name"));
    TEST_ASSERT_FALSE(json_get_boolean(object, "dangling"));

    TEST_ASSERT_FALSE(json_get_boolean(object, "undefined"));
}

void test_get_array(void) {
    JsonArray *array = json_get_array(object, "list");
    TEST_ASSERT_NOT_NULL(array);

    TEST_ASSERT_EQUAL_INT(5, array->size);

    JsonValue *value = array->values;
    TEST_ASSERT_EQUAL_DOUBLE(123, value->number);
    value = value->next;
    TEST_ASSERT_EQUAL_STRING("foobar", value->string);
    value = value->next;
    TEST_ASSERT_EQUAL(true, value->boolean);
    value = value->next;
    TEST_ASSERT_NULL(value->string);
    value = value->next;
    TEST_ASSERT_EQUAL_DOUBLE(5, json_get_number(value->object, "index"));

    TEST_ASSERT_NULL(json_get_array(object, "pi"));
    TEST_ASSERT_NULL(json_get_array(object, "name"));
    TEST_ASSERT_NULL(json_get_array(object, "is_tested"));
    TEST_ASSERT_NULL(json_get_array(object, "is_ignored"));
    TEST_ASSERT_NULL(json_get_array(object, "dangling"));

    TEST_ASSERT_NULL(json_get_array(object, "undefined"));
}

void test_get_object(void) {
    JsonObject *child_object = json_get_child_object(object, "object");
    TEST_ASSERT_NOT_NULL(child_object);

    TEST_ASSERT_EQUAL_DOUBLE(42, json_get_number(child_object, "value"));
}
