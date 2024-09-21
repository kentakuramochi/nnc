/**
 * @file util.h
 * @brief Utilities
 */
#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

/**
 * @brief Allocate a 2-D float memory for a dataset
 *
 * @param[in] size Number of samples in a dataset
 * @param[in] elem_size Number of elements in one sample
 * @return Pointer to the allocated memory, NULL if failed
 */
float **alloc_dataset(const size_t size, const int elem_size);

/**
 * @brief Free a 2-D float memory for a dataset
 *
 * @param[in,out] dataset Pointer to a memory for a dataset
 * @param[in] size Number of samples in a dataset
 */
void free_dataset(float ***dataset, const size_t size);

/**
 * @brief Get an index of the max. element in a vector
 * 
 * @param[in] vector 1-D float array
 * @param[in] size Number of elements of the vector
 * @return Index of the max. element, -1 if failed
 */
int argmax(const float* vector, const int size);

#endif // UTIL_H
