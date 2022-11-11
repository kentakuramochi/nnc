/**
 * @file scnn_mat.c
 * @brief Matrix structure
 * 
 */
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "scnn_mat.h"
#include "scnn_blas.h"

/**
 * @brief Count the number of valid dimension
 * 
 * @param[in]   shape   Matrix shape
 * @param[out]  size    The number of elements
 * @return              The number of valid dimension, 0 if invalid
 */
inline static int count_valid_dim(const int *shape, int *size)
{
    int elems = 1;
    int n_dim = 0;

    // check and count dimension
    bool has_zero_dim = false;
    for (int i = 0; i < 4; i++) {
        if (shape[i] > 0) {
            // fail if 0 has appeared previously
            // e.g.) (1, 0, 1, 1)
            if (has_zero_dim) {
                return 0;
            }
            n_dim++;
            elems *= shape[i];
        } else if (shape[i] < 0) {
            // fail if the dimension is negative
            // e.g.) (-1, 1, 1, 1)
            return 0;
        } else {
            // fail when the first dimension is 0
            // e.g.) (0, 1, 1, 1)
            if (i == 0) {
                return 0;
            }
            has_zero_dim = true;
        }
    }

    *size = elems;

    return n_dim;
}

/**
 * @brief Set the shape to matrix
 * 
 * @param[out]  mat     Matrix
 * @param[in]   shape   Matrix shape
 * @param[in]   n_dim   The number of dimension
 */
inline static void set_mat_shape(scnn_mat *mat, const int *shape, const int n_dim)
{
    // interpolate omitted dimensions
    int shape_idx = n_dim - 4;
    for (int i = 0; i < 4; i++) {
        // omitted dimension is set to 1
        // e.g.) { 2, 3 } -> (1, 1, 2, 3)
        mat->shape[i] = ((shape_idx >= 0) ? shape[shape_idx] : 1);
        shape_idx++;
    }
}

/**
 * @brief Assign data to the matrix
 * 
 * @param[out]  mat     Matrix
 * @param[in]   size    Data size
 * @param[in]   order   Data order
 * @return              Pointer to data array of the matrix, NULL if failed
 */
inline static scnn_dtype *assign_mat_data(scnn_mat *mat, const int size, const scnn_mat_order order)
{
    mat->size   = size;
    mat->order  = order;
    mat->data   = malloc(sizeof(scnn_dtype) * size);

    return mat->data;
}

// Pi constant
static const float PI = 3.141592f;

scnn_mat *scnn_mat_alloc(const int *shape)
{
    if (shape == NULL) {
        return NULL;
    }

    int size;
    int n_dim = count_valid_dim(shape, &size);
    if (n_dim < 1) {
        return NULL;
    }

    scnn_mat *mat = malloc(sizeof(scnn_mat));
    if (mat == NULL) {
        return NULL;
    }

    set_mat_shape(mat, shape, n_dim);

    if (assign_mat_data(mat, size, SCNN_MAT_ORDER_NCHW) == NULL) {
        free(mat);
        mat = NULL;
        return NULL;
    }

    return mat;
}

void scnn_mat_free(scnn_mat **mat)
{
    if ((mat == NULL) || (*mat == NULL)) {
        return;
    }

    free((*mat)->data);
    (*mat)->data = NULL;

    free(*mat);
    *mat = NULL;
}

scnn_mat *scnn_mat_fill(scnn_mat *mat, const scnn_dtype value)
{
    if (mat == NULL) {
        return NULL;
    }

    if ((mat->shape[0] < 1) || (mat->shape[1] < 1) || (mat->shape[2] < 1) || (mat->shape[3] < 1) ||
        (mat->size < 1) ||
        (mat->data == NULL)) {
        return NULL;
    }

    for (int i = 0; i < mat->size; i++) {
        mat->data[i] = value;
    }

    return mat;
}

scnn_mat *scnn_mat_zeros(const int *shape)
{
    scnn_mat *mat = scnn_mat_alloc(shape);
    if (mat == NULL) {
        return NULL;
    }

    return scnn_mat_fill(mat, 0);
}

scnn_mat *scnn_mat_rand(const int *shape)
{
    scnn_mat *mat = scnn_mat_alloc(shape);
    if (mat == NULL) {
        return NULL;
    }

    for (int i = 0; i < mat->size; i++) {
        mat->data[i] = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    }

    return mat;
}

scnn_mat *scnn_mat_randn(const int *shape, const float mean, const float std)
{
    scnn_mat *mat = scnn_mat_alloc(shape);
    if (mat == NULL) {
        return NULL;
    }

    for (int i = 0; i < mat->size; i++) {
        // Box-Muller's method
        float x = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float y = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float z = sqrtf(-2 * logf(x)) * cosf(2 * PI * y);

        mat->data[i] = std * z + mean;
    }

    return mat;
}

scnn_mat *scnn_mat_from_array(const scnn_dtype *array, const int size, const int *shape)
{
    if (array == NULL) {
        return NULL;
    }

    if (size < 1) {
        return NULL;
    }

    if (shape == NULL) {
        return NULL;
    }

    int mat_size;
    int n_dim = count_valid_dim(shape, &mat_size);
    if (n_dim < 1) {
        return NULL;
    }

    if (size != mat_size) {
        return NULL;
    }

    scnn_mat *mat = malloc(sizeof(scnn_mat));
    if (mat == NULL) {
        return NULL;
    }

    set_mat_shape(mat, shape, n_dim);

    if (assign_mat_data(mat, size, SCNN_MAT_ORDER_NCHW) == NULL) {
        free(mat);
        mat = NULL;
        return NULL;
    }

    scnn_scopy(size, array, 1, mat->data, 1);

    return mat;
}
