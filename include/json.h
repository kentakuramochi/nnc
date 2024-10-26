/**
 * @file json.h
 * @brief JSON file operation
 */
#ifndef JSON_H
#define JSON_H

#include <stddef.h>

/**
 * @brief Key-value pair in the JSON object
 * @note Implemented as the doubly linked list
 */
typedef struct KeyValuePair {
    struct KeyValuePair *prev; //!< Previous pair
    struct KeyValuePair *next; //!< Next pair
    char *key;  //!< Key string
    char *value; //!< Value string
} KeyValuePair;

/**
 * @brief JSON object
 * @note Implemented as the doubly linked list
 */
typedef struct JsonObject {
    struct KeyValuePair *kvps; //!< Key-value pair
} JsonObject;

/**
 * @brief Read a JSON object fro a JSON file
 *
 * @param[in] json_file Path to a JSON file
 * @return Pointer to the root JSON object
 */
JsonObject *json_read_file(const char *json_file);

/**
 * @brief Get an integer from the JSON object
 *
 * @param[out] value Integer
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @note value doesn't be changed if the key is not found
 */
void json_get_integer_value(
    int *value, JsonObject *json_object, const char *key
);

/**
 * @brief Get a string from the JSON object
 *
 * @param[out] string String
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @note value doesn't be changed if the key is not found
 */
void json_get_string_value(
    char *string, JsonObject *json_object, const char *key
);

/**
 * @brief Get a floating-point value from the JSON object
 *
 * @param[out] value Floating-point value
 * @param[in] json_object JSON object
 * @param[in] key Key of the object
 * @note value doesn't be changed if the key is not found
 */
void json_get_float_value(
    float *value, JsonObject *json_object, const char *key
);

/**
 * @brief Free JSON object
 * @param[out] json_object JSON object
 */
void json_free_object(JsonObject **json_object);

#endif // JSON_H
