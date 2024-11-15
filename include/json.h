/**
 * @file json.h
 * @brief JSON file operation
 */
#ifndef JSON_H
#define JSON_H

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Wrapper of values in the JSON object
 */
typedef struct JsonValue {
    struct JsonValue *prev; //!< Previous value
    struct JsonValue *next; //!< Next value
    char *string; //!< Value string
} JsonValue;

// Prototype declaration for circular reference
// between JsonKeyValuePair and JsonObject
struct JsonObject;

/**
 * @brief Key-value pair in the JSON object
 */
typedef struct JsonKeyValuePair {
    struct JsonKeyValuePair *prev; //!< Previous pair
    struct JsonKeyValuePair *next; //!< Next pair
    char *key; //!< Key string
    bool has_child; //!< Flag for existence of child object
    union {
        JsonValue *value; //!< Value
        struct JsonObject *child; //!< Child JSON object
    };
} JsonKeyValuePair;

/**
 * @brief JSON object
 */
typedef struct JsonObject {
    JsonKeyValuePair *kvps; //!< Key-value pair
} JsonObject;

/**
 * @brief Read a JSON object fro a JSON file
 *
 * @param[in] json_file Path to a JSON file
 * @return Pointer to the root JSON object
 */
JsonObject *json_read_file(const char *json_file);

/**
 * @brief Get a value from the JSON object
 *
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return Pointer to the JSON value, NULL if failed
 */
JsonValue *json_get_value(
    JsonObject *json_object, const char *key
);

/**
 * @brief Get an integer from the JSON object
 *
 * @param[out] value Integer
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return true Suceed to get the value
 * @return false Fail to get the value
 * @note value doesn't be changed if the key is not found
 */
bool json_get_integer_value(
    int *value, JsonObject *json_object, const char *key
);

/**
 * @brief Get a floating-point value from the JSON object
 *
 * @param[out] value Floating-point value
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return true Suceed to get the value
 * @return false Fail to get the value
 * @note value doesn't be changed if the key is not found
 */
bool json_get_float_value(
    float *value, JsonObject *json_object, const char *key
);

/**
 * @brief Get a string from the JSON object
 *
 * @param[out] string String
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return true Suceed to get the value
 * @return false Fail to get the value
 * @note value doesn't be changed if the key is not found
 */
bool json_get_string_value(
    char *string, JsonObject *json_object, const char *key
);

/**
 * @brief Get a boolean value from the JSON object
 *
 * @param[out] boolean Boolean value
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return true Suceed to get the value
 * @return false Fail to get the value
 * @note value doesn't be changed if the key is not found
 */
bool json_get_boolean_value(
    bool *boolean, JsonObject *json_object, const char *key
);

/**
 * @brief Get a child JSON object from the parent JSON object
 *
 * @param[in] parent_object Parent JSON object
 * @param[in] key Key of the object
 * @return Pointer to the object, NULL if failed
 */
JsonObject *json_get_child_object(JsonObject *parent_object, const char *key);

/**
 * @brief Free JSON object
 * @param[out] json_object JSON object
 */
void json_free_object(JsonObject **json_object);

#endif // JSON_H
