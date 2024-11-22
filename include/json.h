/**
 * @file json.h
 * @brief JSON file operation
 */
#ifndef JSON_H
#define JSON_H

#include <stdbool.h>
#include <stddef.h>

// Prototype declaration for circular reference
// between JsonKeyValuePair and JsonObject
struct JsonObject;

/**
 * @brief Data type of the value
 */
typedef enum {
    JSONDTYPE_NULL, //!< null
    JSONDTYPE_NUMBER, //!< Number
    JSONDTYPE_STRING, //!< String
    JSONDTYPE_BOOLEAN, //!< Boolean
    JSONDTYPE_OBJECT //!< Object
} JsonDType;

/**
 * @brief Wrapper of a value in the JSON object
 */
typedef struct JsonValue {
    struct JsonValue *prev; //!< Previous value
    struct JsonValue *next; //!< Next value
    JsonDType dtype; //!< Data type
    union {
        double number; //!< Number
        char *string; //!< Value string
        bool boolean; //!< Boolean
        struct JsonObject *object; //!< JSON object
    };
} JsonValue;

/**
 * @brief Key-value pair in the JSON object
 */
typedef struct JsonKeyValuePair {
    struct JsonKeyValuePair *prev; //!< Previous pair
    struct JsonKeyValuePair *next; //!< Next pair
    char *key; //!< Key string
    JsonValue *value;
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
JsonValue *json_get_value(JsonObject *json_object, const char *key);

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
 * @brief Free JSON object
 * @param[out] json_object JSON object
 */
void json_free_object(JsonObject **json_object);

#endif // JSON_H
