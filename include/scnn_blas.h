/**
 * @file scnn_blas.h
 * @brief Matrix operations for data array
 * @note  Row-major order
 * 
 */
#ifndef SCNN_BLAS_H
#define SCNN_BLAS_H

/**
 * @brief Flag for transpose of matrix
 * 
 */
typedef enum scnn_blas_transpose {
    SCNN_BLAS_NO_TRANS, //!< No tranpose
    SCNN_BLAS_TRANS     //!< Transpose
} scnn_blas_transpose;

/**
 * @brief General matrix multiplication C = alpha * AB + beta * C 
 * 
 * @param[in] transa    Flag for transpose of matrix A
 * @param[in] transb    Flag for transpose of matrix B
 * @param[in] M         Rows of matrix A
 * @param[in] N         Cols of matrix B
 * @param[in] K         Cols of matrix A, Rows of matrix B
 * @param[in] alpha     Scalar alpha
 * @param[in] A         MxK Matrix A
 * @param[in] lda       Leading dimension of A
 * @param[in] B         KxN Matrix B
 * @param[in] ldb       Leading dimension of B
 * @param[in] beta      Scalar beta
 * @param[in,out] C     MxN Matrix C
 * @param[in] ldc       Leading dimension of C
 * @retval
 */
void scnn_sgemm(const scnn_blas_transpose transa, const scnn_blas_transpose transb,
    const int M, const int N, const int K,
    const float alpha, const float *A, const int lda,
    const float *B, const int ldb,
    const float beta, float *C, const int ldc);

#endif // SCNN_BLAS_H
