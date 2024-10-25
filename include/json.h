/**
 * @file json.h
 * @brief JSON file operation
 */
#ifndef JSON_H
#define JSON_H

#include <stddef.h>

typedef struct JsonObject {
    struct JsonObject *prev;
    struct JsonObject *next;
    struct JsonObject *child;
    struct JsonObject *parent;
    char *key;
    char *value;
} JsonObject;

JsonObject *json_read_file(const char *json_file);

void json_get_integer_value(
    int *value, JsonObject *json_object, const char *key
);

void json_free_object(JsonObject **json_object);

#endif // JSON_H
