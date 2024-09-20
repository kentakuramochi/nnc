/**
 * @file util.h
 * @brief Utilities
 */
#ifndef UTIL_H
#define UTIL_H

/**
 * @brief Allocate a memory for the dataset
 *
 * @param[in] data_num Number of data in a dataset
 * @param[in] data_size Data size in bytes
 * @return Pointer to the allocated memory, NULL if failed
 */
float **alloc_dataset(const int data_num, const int data_size);

/**
 * @brief Free a memory for a dataset
 *
 * @param[in,out] dataset Memory for a dataset
 * @param[in] data_num Number of data in a dataset
 */
void free_dataset(float **dataset, const int data_num);

#endif // UTIL_H
