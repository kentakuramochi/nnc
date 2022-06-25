/**
 * @file scnn_mat.c
 * @brief Matrix structure
 * 
 */
#include <stdlib.h>

#include "scnn_mat.h"

scnn_mat *scnn_mat_alloc(void)
{
    scnn_mat *mat = malloc(sizeof(scnn_mat));
    if (mat == NULL) {
        return NULL;
    }

    mat->n     = 0;
    mat->c     = 0;
    mat->h     = 0;
    mat->w     = 0;
    mat->size  = 0;
    mat->order = SCNN_MAT_ORDER_NCHW;
    mat->data  = NULL;

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

    mat->n    = n;
    mat->c    = c;
    mat->h    = h;
    mat->w    = w;
    mat->size = n * c * h * w;

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

    if ((mat->n < 1) || (mat->c < 1) || (mat->h < 1) || (mat->w < 1) ||
        (mat->size < 1) ||
        (mat->data == NULL)) {
        return NULL;
    }

    for (int i = 0; i < mat->size; i++) {
        mat->data[i] = value;
    }

    return mat;
}
