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
 * @return Pointer to matrix, NULL if failed
 */
scnn_mat *scnn_mat_alloc(void);

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

#endif // SCNN_MAT_H
