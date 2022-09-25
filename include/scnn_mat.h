/**
 * @file scnn_mat.h
 * @brief Matrix structure
 * 
 */
#ifndef SCNN_MAT_H
#define SCNN_MAT_H

/**
 * @brief Data type of matrix element
 * 
 */
typedef float scnn_dtype;

/**
 * @brief Matrix shape
 * 
 */
typedef struct scnn_shape {
    int d[4]; //!< 4-d shape (NCHW order)
} scnn_shape;

/**
 * @brief Data order of matrix element
 * 
 */
typedef enum scnn_mat_order {
    SCNN_MAT_ORDER_NCHW //!< NCHW order
} scnn_mat_order;

/**
 * @brief Matrix structure
 * 
 */
typedef struct scnn_mat {
    int             n;      //!< Batch size N
    int             c;      //!< Channel size C
    int             h;      //!< Height H
    int             w;      //!< Width W
    int             size;   //!< Total size of elements
    scnn_mat_order  order;  //!< Data order
    scnn_dtype      *data;  //!< Data
} scnn_mat;

/**
 * @brief Allocate matrix
 * 
 * @param[in] shape Matrix shape
 * @return          Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_alloc(const scnn_shape shape);

/**
 * @brief Initialize matrix with specified size
 * 
 * @param[in] n Batch size N
 * @param[in] c Channel size C
 * @param[in] h Height H
 * @param[in] w Width W
 * @return      Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_init(scnn_mat *mat, const int n, const int c, const int h, const int w);

/**
 * @brief Free matrix
 * 
 * @param[in,out] mat Pointer to pointer of matrix
 */
void scnn_mat_free(scnn_mat **mat);

/**
 * @brief Fill matrix with specified value
 * 
 * @param[in,out] mat   Pointer to matrix
 * @param[in]     value Filling value
 * @return              Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_fill(scnn_mat *mat, const scnn_dtype value);

/**
 * @brief Copy matrix elements from array
 * 
 * @param[in,out] mat   Pointer to destination matrix
 * @param[in]     array Pointer to source array 
 * @param[in]     size  Num of elements to be copied
 * @return              Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_copy_from_array(scnn_mat *mat, const float *array, const int size);

#endif // SCNN_MAT_H
