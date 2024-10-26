/**
 * @file json.h
 * @brief JSON file operation
 */
#ifndef JSON_H
#define JSON_H

#include <stddef.h>

/**
 * @brief JSON object
 * @note Implemented as the doubly linked list
 */
typedef struct JsonObject {
    struct JsonObject *prev; //!< Previous object
    struct JsonObject *next; //!< Next object
    struct JsonObject *child; //!< Child object
    struct JsonObject *parent; //!< Parent object
    char *key; //!< Key string
    char *value; //!< Value string
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
 * @brief Free JSON object
 * @param[out] json_object JSON object
 */
void json_free_object(JsonObject **json_object);

#endif // JSON_H
