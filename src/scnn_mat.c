/**
 * @file scnn_mat.c
 * @brief Matrix structure
 * 
 */
#include <stdlib.h>

#include "scnn_mat.h"

scnn_mat *scnn_mat_alloc(const int n, const int c, const int h, const int w)
{
    if ((n < 1) || (c < 1) || (h < 1) || (w < 1)) {
        return NULL;
    }

    scnn_mat *mat = malloc(sizeof(scnn_mat));
    if (mat == NULL) {
        goto FREE_MAT;
    }

    mat->n    = n;
    mat->c    = c;
    mat->h    = h;
    mat->w    = w;
    mat->size = n * c * h * w;

    mat->data = malloc(sizeof(scnn_dtype) * n * c * h * w);
    if (mat->data == NULL) {
        goto FREE_MAT_DATA;
    }

    return mat;

FREE_MAT_DATA:
    free(mat->data);
    mat->data = NULL;

FREE_MAT:
    free(mat);
    mat = NULL;

    return NULL;
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
