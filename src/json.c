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

#define BUFFER_SIZE 256

JsonObject *json_read_file(const char *json_file) {
    FILE *fp = fopen(json_file, "r");
    if (fp == NULL) {
        return NULL;
    }

    JsonObject *root_obj = NULL;

    char buffer[BUFFER_SIZE];
    size_t size;
    int depth = 0; // Object's depth
    JsonObject *cur_obj = NULL; // Current object
    while ((size = get_token(buffer, fp, BUFFER_SIZE)) > 0) {
        if (str_equal("{", buffer)) {
            // Start object
            JsonObject *obj = malloc(sizeof(JsonObject));
            obj->prev = NULL;
            obj->next = NULL;
            obj->child = NULL;
            obj->parent = NULL;
            obj->key = NULL;
            obj->value = NULL;

            if (cur_obj == NULL) {
                // If there's no current object, set it to the root
                root_obj = obj;
            } else {
                // Otherwise, append a new object to the next
                cur_obj->next = obj;
                obj->prev = cur_obj;
            }

            cur_obj = obj;
        } else if (str_equal("}", buffer)) {
            // End object
        } else if (str_equal("[", buffer)) {
            // Start array
        } else if (str_equal("]", buffer)) {
            // End array
        } else {
            // Thought as JSON file is written correctly
            // Get a key string
            cur_obj->key = malloc(sizeof(char) * (size + 1));
            strncpy(cur_obj->key, buffer, (size + 1));

            // Get a value string
            size = get_token(buffer, fp, BUFFER_SIZE);
            cur_obj->value = malloc(sizeof(char) * (size + 1));
            strncpy(cur_obj->value, buffer, (size + 1));
        }
    }

    fclose(fp);

    // Return the root object
    return root_obj;
}

void json_get_integer_value(
    int *value, JsonObject *json_object, const char *key
) {
    JsonObject *obj = json_object;

    while (obj != NULL) {
        if (str_equal(key, obj->key)) {
            *value = strtol(obj->value, NULL, 10);
            return;
        }

        obj = obj->next;
    }
}

void json_free_object(JsonObject **json_object) {
    JsonObject *cur_obj = *json_object;

    // Free memories recursively
    while (cur_obj != NULL) {
        free(cur_obj->key);
        cur_obj->key = NULL;
        free(cur_obj->value);
        cur_obj->value = NULL;

        JsonObject *next_obj = cur_obj->next;
        free(cur_obj);
        cur_obj = NULL;

        cur_obj = next_obj;
    }

    // Need to set NULL again?
    *json_object = NULL;
}
