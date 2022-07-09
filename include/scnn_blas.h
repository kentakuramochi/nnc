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
 * @brief Dot product of 2 vectors (transpose of X) * Y
 * 
 * @param[in] n     Length (> 0)
 * @param[in] x     Pointer to vector x
 * @param[in] incx  Stride between elements of x (!= 0, reversal order if negative)
 * @param[in] y     Pointer to vector x
 * @param[in] incy  Stride between elements of y (!= 0, reversal order if negative)
 * @return          Dot product x * y, 0 if failed
 */
float scnn_sdot(const int n, const float *x, const int incx, const float *y, const int incy);

/**
 * @brief Vector addition Y = alpha * X + Y
 * 
 * @param[in] n     Length (> 0)
 * @param[in] alpha Scalar alpha
 * @param[in] x     Pointer to vector x
 * @param[in] incx  Stride between elements of x (!= 0, reversal order if negative)
 * @param[in,out] y Pointer to vector y
 * @param[in] incy  Stride between elements of y (!= 0, reversal order if negative)
 */
void scnn_saxpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy);

/**
 * @brief General matrix multiplication C = alpha * AB + beta * C 
 * 
 * @param[in] transa    Flag for transpose of matrix A
 * @param[in] transb    Flag for transpose of matrix B
 * @param[in] M         Rows of matrix A (> 0)
 * @param[in] N         Cols of matrix B (> 0)
 * @param[in] K         Cols of matrix A, Rows of matrix B (> 0)
 * @param[in] alpha     Scalar alpha
 * @param[in] A         Pointer to MxK matrix A
 * @param[in] lda       Leading dimension of A (> 0)
 * @param[in] B         Pointer to KxN matrix B
 * @param[in] ldb       Leading dimension of B (> 0)
 * @param[in] beta      Scalar beta
 * @param[in,out] C     Pointer to MxN matrix C
 * @param[in] ldc       Leading dimension of C (> 0)
 * @retval
 */
void scnn_sgemm(const scnn_blas_transpose transa, const scnn_blas_transpose transb,
    const int M, const int N, const int K,
    const float alpha, const float *A, const int lda,
    const float *B, const int ldb,
    const float beta, float *C, const int ldc);

#endif // SCNN_BLAS_H
