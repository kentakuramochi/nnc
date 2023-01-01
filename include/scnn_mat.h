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
 * @brief Max dimension of matrix
 * 
 */
#define SCNN_MAT_DIM 4

/**
 * @brief Utility to indicate 4-d matrix shape
 * 
 */
#define scnn_shape(...) (int[SCNN_MAT_DIM]){ __VA_ARGS__ }

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
    int             shape[SCNN_MAT_DIM];    //!< Matrix shape
    int             size;                   //!< Total size of elements
    scnn_mat_order  order;                  //!< Data order
    scnn_dtype      *data;                  //!< Data
} scnn_mat;

/**
 * @brief Allocate matrix
 * 
 * @param[in] shape Matrix shape
 * @return          Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_alloc(const int *shape);

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
 * @brief Allocate zero-filled matrix
 * 
 * @param[in] shape Matrix shape
 * @return          Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_zeros(const int *shape);

/**
 * @brief Allocate matrix filled with random values from uniform distribution over [0.0, 1.0)
 * 
 * @param[in] shape Matrix shape
 * @return          Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_rand(const int *shape);

/**
 * @brief Allocate matrix filled with random values from normal distribution 
 * 
 * @param[in] shape Matrix shape
 * @param[in] mean  Mean
 * @param[in] std   Standard deviation
 * @return          Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_randn(const int *shape, const float mean, const float std);

/**
 * @brief Allocate matrix from data array with specified shape
 * 
 * @param[in] array Source array
 * @param[in] size  Size of source array
 * @param[in] shape Matrix shape
 * @return          Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_from_array(const scnn_dtype *array, const int size, const int *shape);

#endif // SCNN_MAT_H
