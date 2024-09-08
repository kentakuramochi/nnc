/**
 * @file random.c
 * @brief Simple PRNG
*/
#include "random.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * @brief Pi value
*/
#define PI 3.14159265

float rand_uniform(void) {
    return (float)rand() / ((uint32_t)RAND_MAX + 1);
}

float rand_norm(const float mean, const float stddev) {
    // Generate a random value by the Box-Muller method
    float u1 = rand_uniform();
    float u2 = rand_uniform();

    // OR: sqrtf(-2 * logf(u1)) * sinf(2 * PI * u2);
    float x = sqrtf(-2 * logf(u1)) * cosf(2 * PI * u2);

    return mean + x * stddev;
}
