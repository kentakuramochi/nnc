/**
 * @file scnn_mat.c
 * @brief Matrix structure
 * 
 */
#include <stdlib.h>
#include <stdbool.h>

#include "scnn_mat.h"
#include "scnn_blas.h"

scnn_mat *scnn_mat_alloc(const scnn_shape shape)
{
    scnn_mat *mat = malloc(sizeof(scnn_mat));
    if (mat == NULL) {
        return NULL;
    }

    // count num of dimension
    int  n_dim = 0;
    bool has_dim_zero = false;
    for (int i = 0; i < 4; i++) {
        if (shape.d[i] > 0) {
            if (has_dim_zero) {
                return NULL;
            }
            n_dim++;
        } else if (shape.d[i] < 0) {
            return NULL;
        } else { // zero
            if (n_dim == 0) {
                // fail when all zero
                return NULL;
            }
            has_dim_zero = true;
        }
    }
    // set 4-d shape with considering with omitted dimension
    // omitted dimenstion is set to 1
    int new_shape[4] = { 0 };
    int size = 1;
    int shape_idx = n_dim - 4;
    for (int i = 0; i < 4; i++) {
        new_shape[i] = ((shape_idx >= 0) ? shape.d[shape_idx] : 1);
        size *= new_shape[i];
        shape_idx++;
    }

    mat->shape.d[0] = new_shape[0];
    mat->shape.d[1] = new_shape[1];
    mat->shape.d[2] = new_shape[2];
    mat->shape.d[3] = new_shape[3];
    mat->size       = size;
    mat->order      = SCNN_MAT_ORDER_NCHW;
    mat->data       = malloc(sizeof(float) * size);

    return mat;
}

scnn_mat *scnn_mat_init(scnn_mat *mat, const int n, const int c, const int h, const int w)
{
    if (mat == NULL) {
        return NULL;
    }

    if ((n < 1) || (c < 1) || (h < 1) || (w < 1)) {
        return NULL;
    }

    mat->shape.d[0] = n;
    mat->shape.d[1] = c;
    mat->shape.d[2] = h;
    mat->shape.d[3] = w;
    mat->size       = n * c * h * w;

    mat->data = malloc(sizeof(scnn_dtype) * n * c * h * w);
    if (mat->data == NULL) {
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

    if ((mat->shape.d[0] < 1) || (mat->shape.d[1] < 1) || (mat->shape.d[2] < 1) || (mat->shape.d[3] < 1) ||
        (mat->size < 1) ||
        (mat->data == NULL)) {
        return NULL;
    }

    for (int i = 0; i < mat->size; i++) {
        mat->data[i] = value;
    }

    return mat;
}

scnn_mat *scnn_mat_copy_from_array(scnn_mat *mat, const float *array, const int size)
{
    if ((mat == NULL) || (array == NULL)) {
        return NULL;
    }

    if (size < 1) {
        return NULL;
    }

    scnn_scopy(size, array, 1, mat->data, 1);

    return mat;
}
