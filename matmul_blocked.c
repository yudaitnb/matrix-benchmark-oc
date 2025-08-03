#include <stdint.h>
#include <stdlib.h>

void matmul_blocked_c(const double* restrict A, const double* restrict B, double* restrict C, int N, int block_size) {
    int i, j, k, ii, jj, kk;

    for (ii = 0; ii < N; ii += block_size) {
        for (jj = 0; jj < N; jj += block_size) {
            for (kk = 0; kk < N; kk += block_size) {
                int i_max = ii + block_size < N ? ii + block_size : N;
                int j_max = jj + block_size < N ? jj + block_size : N;
                int k_max = kk + block_size < N ? kk + block_size : N;

                for (i = ii; i < i_max; ++i) {
                    for (k = kk; k < k_max; ++k) {
                        double a_ik = A[i * N + k];
                        for (j = jj; j < j_max; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
