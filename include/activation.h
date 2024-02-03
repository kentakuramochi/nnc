/**
 * @file activation.h
 * @brief Activation function
 *
 */
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stddef.h>

/**
 * @brief Sigmoid function
 * 
 * @param x[in]     Input array
 * @param y[out]    Output array
 * @param size[in]  Number of input elements
 * @return float*   Sigmoid
 */
float *sigmoid(const float *x, float *y, const size_t size);

/**
 * @brief Softmax function
 * 
 * @param x[in]     Input array
 * @param y[out]    Output array
 * @param size[in]  Number of input elements
 * @return float*   Softmax
 */
float *softmax(const float *x, float *y, const size_t size);

#endif // ACTIVATION_H
