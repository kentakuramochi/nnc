#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string.h>

#define FLOAT_ARRAY(...) (float[]){ __VA_ARGS__ }

#define FLOAT_ZEROS(size) (float[(size)]){ 0 }

static inline void copy_array(float *dst, const float *src, const size_t size) {
    memcpy(dst, src, size);
}

#define COPY_ARRAY(dst, ...) copy_array((dst), __VA_ARGS__, sizeof(__VA_ARGS__))

#endif // TEST_UTILS_H
