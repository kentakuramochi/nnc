/**
 * @file json.h
 * @brief JSON file operation
 */
#ifndef JSON_H
#define JSON_H

#include <stdbool.h>
#include <stddef.h>

// Prototype declaration for circular reference
struct JsonObject;

/**
 * @brief Data type of the value
 */
typedef enum {
    JSONDTYPE_NULL, //!< null
    JSONDTYPE_NUMBER, //!< Number
    JSONDTYPE_STRING, //!< String
    JSONDTYPE_BOOLEAN, //!< Boolean
    JSONDTYPE_OBJECT, //!< Object
    JSONDTYPE_ARRAY //!< Array
} JsonDType;

/**
 * @brief Wrapper of a value in the JSON object
 */
typedef struct JsonValue {
    JsonDType dtype; //!< Data type
    size_t size; //!< Size
    union {
        double number; //!< Number
        char *string; //!< Value string
        bool boolean; //!< Boolean
        struct JsonObject *object; //!< JSON object
        struct JsonValue *values; //!< Array
    };
} JsonValue;

/**
 * @brief Key-value pair in the JSON object
 */
typedef struct JsonKeyValuePair {
    struct JsonKeyValuePair *prev; //!< Previous pair
    struct JsonKeyValuePair *next; //!< Next pair
    char *key; //!< Key string
    JsonValue *value; //!< Value
} JsonKeyValuePair;

/**
 * @brief JSON object
 */
typedef struct JsonObject {
    JsonKeyValuePair *kvps; //!< Key-value pair
} JsonObject;

/**
 * @brief Read a JSON object from a JSON file
 *
 * @param[in] json_file Path to a JSON file
 * @return Pointer to the root JSON object
 */
JsonObject *json_read_file(const char *json_file);

/**
 * @brief Get a number from the JSON object
 *
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return Number
 * @retval INFINITY Fail to get a number
 */
double json_get_number(JsonObject *json_object, const char *key);

/**
 * @brief Get a string from the JSON object
 *
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return Pointer to NULL-terminated string
 * @retval NULL Fail to get a string
 */
char *json_get_string(JsonObject *json_object, const char *key);

/**
 * @brief Get a boolean from the JSON object
 *
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return Boolean [true|false]
 * @retval false Fail to get a boolean
 */
bool json_get_boolean(JsonObject *json_object, const char *key);

/**
 * @brief Get a child JSON object from the parent JSON object
 *
 * @param[in] parent_object Parent JSON object
 * @param[in] key Key of the object
 * @return Pointer to the object, NULL if failed
 */
JsonObject *json_get_child_object(JsonObject *parent_object, const char *key);

/**
 * @brief Get an array from the JSON object
 *
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @return Pointer to the JSON array, NULL if failed
 */
JsonValue *json_get_array(JsonObject *json_object, const char *key);

/**
 * @brief Free JSON object
 * @param[out] json_object JSON object
 */
void json_free_object(JsonObject **json_object);

#endif // JSON_H
