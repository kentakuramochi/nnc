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

        // Terminate the toke if the length reaches to the max length
        if (token_length == (buf_size - 1)) {
            buffer[token_length] = '\0';
            break;
        }
    }

    return token_length;
}

//!< Size of the temporal buffer
#define BUFFER_SIZE 256

JsonObject *json_read_file(const char *json_file) {
    FILE *fp = fopen(json_file, "r");
    if (fp == NULL) {
        return NULL;
    }

    JsonObject *cur_obj = NULL;
    JsonKeyValuePair *cur_kvp = NULL;

    char buffer[BUFFER_SIZE];
    size_t size;
    while ((size = get_token(buffer, fp, BUFFER_SIZE)) > 0) {
        if (str_equal("{", buffer)) {
            // Start object
            JsonObject *obj = malloc(sizeof(JsonObject));
            obj->kvps = NULL;

            // If there's no current object, set it to the root
            if (cur_obj == NULL) {
                cur_obj = obj;
            }
        } else if (str_equal("}", buffer)) {
            // End object
        } else {
            JsonKeyValuePair *kvp = malloc(sizeof(JsonKeyValuePair));
            kvp->prev = NULL;
            kvp->next = NULL;
            kvp->key = NULL;
            kvp->value = NULL;

            // Append the kvp to the current one
            if (cur_kvp != NULL) {
                cur_kvp->next = kvp;
                kvp->prev = cur_kvp;
            }
            cur_kvp = kvp;

            // Set the kvp to the object if not set
            if (cur_obj->kvps == NULL) {
                cur_obj->kvps = cur_kvp;
            }

            // Thought as JSON file is written correctly
            // Get a key string
            cur_kvp->key = malloc(sizeof(char) * (size + 1));
            strncpy(cur_kvp->key, buffer, (size + 1));

            // Get a value
            size = get_token(buffer, fp, BUFFER_SIZE);
            JsonValue *value = malloc(sizeof(JsonValue));
            value->string = malloc(sizeof(char) * (size + 1));
            strncpy(value->string, buffer, (size + 1));
            cur_kvp->value = value;
        }
    }

    fclose(fp);

    return cur_obj;
}

void json_get_integer_value(
    int *value, JsonObject *json_object, const char *key
) {
    JsonKeyValuePair *kvp = json_object->kvps;

    while (kvp != NULL) {
        if (str_equal(key, kvp->key)) {
            *value = strtol(kvp->value->string, NULL, 10);
            return;
        }
        kvp = kvp->next;
    }
}

void json_get_string_value(
    char *string, JsonObject *json_object, const char *key
) {
    JsonKeyValuePair *kvp = json_object->kvps;

    while (kvp != NULL) {
        if (str_equal(key, kvp->key)) {
            // Copy +1 length to terminate with NULL
            strncpy(string, kvp->value->string, (strlen(kvp->value->string) + 1));
            return;
        }
        kvp = kvp->next;
    }
}

void json_get_float_value(
    float *value, JsonObject *json_object, const char *key
) {
    JsonKeyValuePair *kvp = json_object->kvps;

    while (kvp != NULL) {
        if (str_equal(key, kvp->key)) {
            *value = strtof(kvp->value->string, NULL);
            return;
        }
        kvp = kvp->next;
    }
}

void json_get_boolean_value(
    bool *boolean, JsonObject *json_object, const char *key
) {
    JsonKeyValuePair *kvp = json_object->kvps;

    while (kvp != NULL) {
        if (str_equal(key, kvp->key)) {
            if (str_equal("true", kvp->value->string)) {
                *boolean = true;
            } else if (str_equal("false", kvp->value->string)) {
                *boolean = false;
            }
            return;
        }
        kvp = kvp->next;
    }
}

void json_free_object(JsonObject **json_object) {
    JsonKeyValuePair *kvp = (*json_object)->kvps;

    // Free memories sequentially
    while (kvp != NULL) {
        free(kvp->key);
        kvp->key = NULL;

        free(kvp->value->string);
        kvp->value->string = NULL;

        free(kvp->value);
        kvp->value = NULL;

        JsonKeyValuePair *next_kvp = kvp->next;
        free(kvp);
        kvp = NULL;

        kvp = next_kvp;
    }

    free(*json_object);
    *json_object = NULL;
}
