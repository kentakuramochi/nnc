/**
 * @file util.c
 * @brief Utilities
 */
#include "util.h"

#include <stdlib.h>

float **alloc_dataset(const int data_num, const int data_size) {
    float **dataset = malloc(sizeof(float*) * data_num);
    if (dataset == NULL) {
        return NULL;
    }

    for (int i = 0; i < data_num; i++) {
        dataset[i] = malloc(sizeof(float) * data_size);
        if (dataset[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(dataset[i]);
                dataset[i] = NULL;
            }
            free(dataset);
            dataset = NULL;
            return NULL;
        }
    }

    return dataset;
}

void free_dataset(float **dataset, const int data_num) {
    if (dataset == NULL) {
        return;
    }

    for (int i = 0; i < data_num; i++) {
        free(dataset[i]);
        dataset[i] = NULL;
    }
    
    free(dataset);
}
