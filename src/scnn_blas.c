/**
 * @file scnn_blas.c
 * @brief Matrix operations for data array
 * 
 */
#include "scnn_blas.h"

#include <stddef.h>

void scnn_sgemm(const scnn_blas_transpose transa, const scnn_blas_transpose transb,
    const int M, const int N, const int K,
    const float alpha, const float *A, const int lda,
    const float *B, const int ldb,
    const float beta, float *C, const int ldc)
{
    if ((A == NULL) || (B == NULL) || (C == NULL)) {
        return;
    }
    if ((M < 1) || (N < 1) || (K < 1) || (lda < 1) || (ldb < 1) || (ldc < 1)) {
        return;
    }

    if (transa == SCNN_BLAS_NO_TRANS) {
        if (transb == SCNN_BLAS_NO_TRANS) {
            // a: NO_TRANS, b: NO_TRANS
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += A[i * lda + k] * B[k * ldb + j];
                    }
                    C[i * ldc + j] *= beta;
                    C[i * ldc + j] += alpha * sum;
                }
            }
        } else {
            // a: NO_TRANS, b: TRANS
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += A[i * lda + k] * B[j * ldb + k];
                    }
                    C[i * ldc + j] *= beta;
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    } else {
        if (transb == SCNN_BLAS_NO_TRANS) {
            // a: TRANS, b: NO_TRANS
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += A[k * lda + i] * B[k * ldb + j];
                    }
                    C[i * ldc + j] *= beta;
                    C[i * ldc + j] += alpha * sum;
                }
            }
        } else {
            // a: TRANS, b: TRANS
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += A[k * lda + i] * B[j * ldb + k];
                    }
                    C[i * ldc + j] *= beta;
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}
