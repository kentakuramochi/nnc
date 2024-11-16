/**
 * @file json.c
 * @brief JSON file operation
 */
#include "json.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Check equality of 2 strings
 *
 * @param[in] s1 NULL-terminated string #1
 * @param[in] s2 NULL-terminated string #2
 * @return true if s1 == s2, otherwise false
 */
static inline bool str_equal(const char *s1, const char *s2) {
    return !strcmp(s1, s2) ? true : false;
}

/**
 * @brief Get a JSON token from a stream
 *
 * @param[out] buffer Buffer to store a NULL-terminated token
 * @param[in] fp Input stream
 * @param[in] buf_size Size of the buffer
 * @return Length of the token string
 * @note Implementation is incomplete, DOES NOT conform to the JSON specification
 */
static size_t get_token(char *buffer, FILE *fp, const size_t buf_size) {
    size_t token_length = 0;

    bool in_string = false;
    char c;
    while ((c = fgetc(fp)) != EOF) {
        if (in_string) {
            // In string, terminate the token by double quote
            // Escape sequense is not considered
            if (c == '"') {
                in_string = false;
                buffer[token_length] = '\0';
                break;
            }
        } else {
            // Terminate the token by JSON whitespace
            if ((c == ' ') || (c == '\n') || (c == '\r') || (c == '\t')) {
                if (token_length > 0) {
                    buffer[token_length] = '\0';
                    break;
                }
                continue;
            }

            if ((c == ':') || (c == ',')) {
                // Terminate the token by colon or comma
                // Escape sequense is not considered
                if (token_length > 0) {
                    buffer[token_length] = '\0';
                    break;
                }
                // Skip if token string hasn't get
                continue;
            }

            // Start string by double quote
            if (c == '"') {
                in_string = true;
                continue;
            }
        }

        buffer[token_length] = c;
        token_length++;

        // Terminate the token if the length reaches to the max length
        if (token_length == (buf_size - 1)) {
            buffer[token_length] = '\0';
            break;
        }
    }

    return token_length;
}

/**
 * @brief Allocate a JSON key-value pair
 *
 * @return Pointer to the allocated JSON key-value pair
 */
static JsonKeyValuePair *alloc_json_key_value_pair(void) {
    JsonKeyValuePair *kvp = malloc(sizeof(JsonKeyValuePair));
    kvp->prev = NULL;
    kvp->next = NULL;
    kvp->key = NULL;
    kvp->value = NULL;

    return kvp;
}

/**
 * @brief Allocate a JSON key
 *
 * @param[in] buffer Bufer which stores a token in the JSON file
 * @param[in] token_size Size of the token
 * @return Pointer to the allocated JSON key
 */
static char *alloc_json_key(
    const char *buffer, const size_t token_size
) {
    char *key = malloc(sizeof(char) * (token_size + 1));
    strncpy(key, buffer, (token_size + 1));

    return key;
}

/**
 * @brief Allocate a JSON value
 *
 * @param[in] buffer Bufer which stores a token in the JSON file
 * @param[in] token_size Size of the token
 * @return Pointer to the allocated JSON value
 */
static JsonValue *alloc_json_value(
    const char *buffer, const size_t token_size
) {
    JsonValue *value = malloc(sizeof(JsonValue));
    value->prev = NULL;
    value->next = NULL;
    value->is_object = false;
    value->string = malloc(sizeof(char) * (token_size + 1));
    strncpy(value->string, buffer, (token_size + 1));

    return value;
}

/**
 * @brief Allocate an array of JSON values
 *
 * @param[in] fp Input file stream
 * @param[in] buffer Bufer which token in the JSON file will be stored in
 * @param[in] buffer_size Size of the buffer
 * @return Pointer to the allocated an array of JSON values
 */
static JsonValue *alloc_json_array(
    FILE *fp, char *buffer, const size_t buffer_size
) {
    JsonValue *head = NULL;
    JsonValue *value = NULL;
    size_t size = 0;
    while ((size = get_token(buffer, fp, buffer_size)) > 0) {
        if (str_equal(buffer, "]")) {
            // End of the array
            break;
        }

        if (head == NULL) {
            value = alloc_json_value(buffer, size);
            head = value;
        } else {
            // Append a new value to the current one
            JsonValue *new_value = alloc_json_value(buffer, size);
            value->next = new_value;
            new_value->prev = value;

            value = value->next;
        }
    }

    return head;
}

/**
 * @brief Allocate a JSON object
 *
 * @param[in] fp Input file stream
 * @param[in] buffer Bufer which token in the JSON file will be stored in
 * @param[in] buffer_size Size of the buffer
 * @return Pointer to the allocated JSON object
 */
static JsonObject *alloc_json_object(
    FILE *fp, char *buffer, const size_t buffer_size
) {
    JsonObject *object = malloc(sizeof(JsonObject));

    JsonKeyValuePair *kvp = NULL;
    size_t size = 0;
    while ((size = get_token(buffer, fp, buffer_size)) > 0) {
        if (str_equal(buffer, "}")) {
            // End of the object
            break;
        }

        if (kvp == NULL) {
            kvp = alloc_json_key_value_pair();
            object->kvps = kvp;
        } else {
            // Append a new key-value pair to the current one
            JsonKeyValuePair *new_kvp = alloc_json_key_value_pair();
            kvp->next = new_kvp;
            new_kvp->prev = kvp;

            kvp = new_kvp;
        }

        // Get a key
        kvp->key = alloc_json_key(buffer, size);

        // Get a value/object
        size = get_token(buffer, fp, buffer_size);
        if (str_equal(buffer, "{")) {
            // If '{' is read, get succeeding tokens an object
            kvp->value = malloc(sizeof(JsonValue));
            kvp->value->prev = NULL;
            kvp->value->next = NULL;
            kvp->value->object = alloc_json_object(fp, buffer, buffer_size);
            kvp->value->is_object = true;
        } else if (str_equal(buffer, "[")) {
            // If '[' is read, get succeeding tokens as an array
            kvp->value = alloc_json_array(fp, buffer, buffer_size);
        } else {
            // Otherwise, get succeeding tokens as a single value
            kvp->value = alloc_json_value(buffer, size);
        }
    }

    return object;
}

//!< Size of the temporal buffer
#define BUFFER_SIZE 256

JsonObject *json_read_file(const char *json_file) {
    FILE *fp = fopen(json_file, "r");
    if (fp == NULL) {
        return NULL;
    }

    JsonObject *object = NULL;
    char buffer[BUFFER_SIZE];
    size_t size = 0;
    while ((size = get_token(buffer, fp, BUFFER_SIZE)) > 0) {
        if (str_equal("{", buffer)) {
            object = alloc_json_object(fp, buffer, BUFFER_SIZE);
        }
        // Currently ignore non-object tokens in the root
    }

    fclose(fp);

    return object;
}

JsonValue *json_get_value(
    JsonObject *json_object, const char *key
) {
    JsonKeyValuePair *kvp = json_object->kvps;

    while (kvp != NULL) {
        if (str_equal(key, kvp->key)) {
            return kvp->value;
        }
        kvp = kvp->next;
    }

    return NULL;
}

bool json_get_integer_value(
    int *value, JsonObject *json_object, const char *key
) {
    JsonValue *jsonValue;
    if ((jsonValue = json_get_value(json_object, key)) == NULL) {
        return false;
    }

    if (str_equal("null", jsonValue->string)) {
        return false;
    }

    *value = strtol(jsonValue->string, NULL, 10);

    return true;
}

bool json_get_float_value(
    float *value, JsonObject *json_object, const char *key
) {
    JsonValue *jsonValue;
    if ((jsonValue = json_get_value(json_object, key)) == NULL) {
        return false;
    }

    if (str_equal("null", jsonValue->string)) {
        return false;
    }

    *value = strtof(jsonValue->string, NULL);

    return true;
}

bool json_get_string_value(
    char *string, JsonObject *json_object, const char *key
) {
    JsonValue *jsonValue;
    if ((jsonValue = json_get_value(json_object, key)) == NULL) {
        return false;
    }

    if (str_equal("null", jsonValue->string)) {
        return false;
    }

    // Copy +1 length to terminate with NULL
    strncpy(string, jsonValue->string, (strlen(jsonValue->string) + 1));

    return true;
}

bool json_get_boolean_value(
    bool *boolean, JsonObject *json_object, const char *key
) {
    JsonValue *jsonValue;
    if ((jsonValue = json_get_value(json_object, key)) == NULL) {
        return false;
    }

    if (str_equal("null", jsonValue->string)) {
        return false;
    }

    if (str_equal("true", jsonValue->string)) {
        *boolean = true;
    } else if (str_equal("false", jsonValue->string)) {
        *boolean = false;
    }

    return true;
}

JsonObject *json_get_child_object(JsonObject *parent_object, const char *key) {
    JsonKeyValuePair *kvp = parent_object->kvps;

    while (kvp != NULL) {
        if (str_equal(key, kvp->key)) {
            if (kvp->value->is_object) {
                return kvp->value->object;
            }
        }
        kvp = kvp->next;
    }

    return NULL;
}

void json_free_object(JsonObject **json_object) {
    JsonKeyValuePair *kvp = (*json_object)->kvps;

    // Free key-value pairs sequentially
    while (kvp != NULL) {
        free(kvp->key);
        kvp->key = NULL;

        // Free values sequentially
        JsonValue *value = kvp->value;
        while (value != NULL) {
            if (value->is_object) {
                // If there's a child object, free recursively
                json_free_object(&value->object);
            } else {
                free(value->string);
                value->string = NULL;
            }

            JsonValue *next = value->next;

            free(value);
            value = NULL;

            value = next;
        }

        JsonKeyValuePair *next_kvp = kvp->next;
        free(kvp);
        kvp = NULL;

        kvp = next_kvp;
    }

    free(*json_object);
    *json_object = NULL;
}
