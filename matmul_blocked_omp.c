#include <omp.h>

void matmul_blocked_omp_c(const double* restrict A, const double* restrict B, double* restrict C, int N, int block_size) {
    int i, j, k, ii, jj, kk;

    #pragma omp parallel for collapse(2) private(ii, jj, kk, i, j, k) schedule(static)
    for (ii = 0; ii < N; ii += block_size) {
        for (jj = 0; jj < N; jj += block_size) {
            for (kk = 0; kk < N; kk += block_size) {
                int i_max = ii + block_size < N ? ii + block_size : N;
                int j_max = jj + block_size < N ? jj + block_size : N;
                int k_max = kk + block_size < N ? kk + block_size : N;

                for (i = ii; i < i_max; ++i) {
                    for (j = jj; j < j_max; ++j) {
                        double sum = C[i * N + j];
                        #pragma omp simd reduction(+:sum)
                        for (k = kk; k < k_max; ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}
