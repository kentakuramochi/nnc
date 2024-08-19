#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string.h>

#define TEST_UTIL_FLOAT_ARRAY(...) (float[]){ __VA_ARGS__ }

#define TEST_UTIL_FLOAT_ZEROS(size) (float[(size)]){ 0 }

static inline void test_util_copy_array(float *dst, const float *src, const size_t size) {
    memcpy(dst, src, size);
}

#define TEST_UTIL_COPY_ARRAY(dst, ...) test_util_copy_array((dst), __VA_ARGS__, sizeof(__VA_ARGS__))

#endif // TEST_UTILS_H
