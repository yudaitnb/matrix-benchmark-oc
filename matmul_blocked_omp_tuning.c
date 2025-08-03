#include <omp.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>
#include <assert.h>

#define MICRO_BLOCK 4
#define MICRO_M 8
#define MICRO_N 4

// アライメント検証
static inline void assert_aligned(const void* ptr, size_t alignment, const char* name) {
    if (((uintptr_t)ptr) % alignment != 0) {
        fprintf(stderr, "⛔ %s is not aligned to %zu bytes: address=%p\n", name, alignment, ptr);
        assert(0 && "Memory alignment error");
    }
}

// Bを転置する（キャッシュ効率向上）
void transpose_for_b_kj(const double* B, double* B_T, int N, int stride, int block_size) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int kk = 0; kk < N; kk += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            int k_max = (kk + block_size < N) ? kk + block_size : N;
            int j_max = (jj + block_size < N) ? jj + block_size : N;
            for (int k = kk; k < k_max; ++k)
                for (int j = jj; j < j_max; ++j)
                    B_T[k * stride + j] = B[j * stride + k];
        }
    }
}

// ベクトルFMA計算（1ピース）
static inline void fma_block(__m256d* c, const double* A, const __m256d b, int p, int stride) {
    for (int i = 0; i < MICRO_M; ++i) {
        __m256d a = _mm256_set1_pd(A[i * stride + p]);
        c[i] = _mm256_fmadd_pd(a, b, c[i]);
    }
}

// ループアンローリング
#define UNROLL1(ACTION)  ACTION(0)
#define UNROLL2(ACTION)  UNROLL1(ACTION)  ACTION(1)
#define UNROLL3(ACTION)  UNROLL2(ACTION)  ACTION(2)
#define UNROLL4(ACTION)  UNROLL3(ACTION)  ACTION(3)
#define UNROLL5(ACTION)  UNROLL4(ACTION)  ACTION(4)
#define UNROLL6(ACTION)  UNROLL5(ACTION)  ACTION(5)
#define UNROLL7(ACTION)  UNROLL6(ACTION)  ACTION(6)
#define UNROLL8(ACTION)  UNROLL7(ACTION)  ACTION(7)

#define DISPATCH_UNROLL(N, ACTION) UNROLL##N(ACTION)
#define FMA_AT_P(p)                                \
    {                                              \
        __m256d b = _mm256_load_pd(&B_T[(p) * stride]); \
        fma_block(c, A, b, (p), stride);           \
    }

static inline void microkernel_8x4_unrolled_dynamic(
    const double* A,
    const double* B_T,
    double* C,
    int stride,
    int K,         // 内積の長さ（列×行）
    int M_actual   // 実際の行数（MICRO_M 以下）
) {
    __m256d c[MICRO_M];
    for (int i = 0; i < M_actual; ++i)
        c[i] = _mm256_load_pd(&C[i * stride]);

    for (int p = 0; p < K; ++p) {
        __m256d b = _mm256_load_pd(&B_T[p * stride]);
        for (int i = 0; i < M_actual; ++i) {
            __m256d a = _mm256_set1_pd(A[i * stride + p]);
            c[i] = _mm256_fmadd_pd(a, b, c[i]);
        }
    }

    for (int i = 0; i < M_actual; ++i)
        _mm256_store_pd(&C[i * stride], c[i]);
}


// メインマトリクス積関数
// メインマトリクス積関数（列端数はスカラー fallback）
void matmul_blocked_omp_tuning_c(
    const double* restrict A,
    const double* restrict B,
    double* restrict C,
    int N,
    int stride,
    int block_size
) {
    double* B_T;
    if (posix_memalign((void**)&B_T, 64, sizeof(double) * stride * stride) != 0 || B_T == NULL) {
        fprintf(stderr, "Failed to allocate memory for B_T\n");
        return;
    }

    transpose_for_b_kj(B, B_T, N, stride, block_size);

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int jj = 0; jj < N; jj += block_size) {
        for (int ii = 0; ii < N; ii += block_size) {
            for (int kk = 0; kk < N; kk += block_size) {
                int i_max = (ii + block_size < N) ? ii + block_size : N;
                int j_max = (jj + block_size < N) ? jj + block_size : N;
                int k_max = (kk + block_size < N) ? kk + block_size : N;

                // i方向の端数処理（動的マイクロカーネル）
                int i_tail = i_max % MICRO_M;
                if (i_tail > 0) {
                    for (int i = i_max - i_tail; i < i_max; i += i_tail) {
                        // j方向の本体（AVX2処理）
                        for (int j = jj; j <= j_max - MICRO_N; j += MICRO_N) {
                            for (int k = kk; k < k_max; k += MICRO_N) {
                                int k_tail = k_max - k;
                                int k_chunk = (k_tail >= MICRO_N) ? MICRO_N : k_tail;

                                // ⏩ Prefetch for A and B_T
                                int prefetch_k = k + 4;
                                if (prefetch_k < k_max) {
                                    _mm_prefetch((const char*)(&A[i * stride + prefetch_k]), _MM_HINT_T0);
                                    _mm_prefetch((const char*)(&B_T[prefetch_k * stride + j]), _MM_HINT_T0);
                                }

                                microkernel_8x4_unrolled_dynamic(
                                    &A[i * stride + k],
                                    &B_T[k * stride + j],
                                    &C[i * stride + j],
                                    stride,
                                    k_chunk,
                                    i_tail
                                );
                            }
                        }

                        // j方向端数（スカラー fallback）
                        int j_tail = j_max % MICRO_N;
                        if (j_tail > 0) {
                            for (int j = j_max - j_tail; j < j_max; ++j) {
                                for (int ii_inner = 0; ii_inner < i_tail; ++ii_inner) {
                                    double sum = 0.0;
                                    for (int k = kk; k < k_max; ++k) {
                                        sum += A[(i + ii_inner) * stride + k] * B_T[k * stride + j];
                                    }
                                    C[(i + ii_inner) * stride + j] = sum;
                                }
                            }
                        }
                    }
                }

                // 本体部分（AVX2 8x4マイクロカーネル）
                for (int i = ii; i <= i_max - MICRO_M; i += MICRO_M) {
                    for (int j = jj; j <= j_max - MICRO_N; j += MICRO_N) {
                        for (int k = kk; k < k_max; k += MICRO_N) {
                            int k_tail = k_max - k;
                            int k_chunk = (k_tail >= MICRO_N) ? MICRO_N : k_tail;

                            microkernel_8x4_unrolled_dynamic(
                                &A[i * stride + k],
                                &B_T[k * stride + j],
                                &C[i * stride + j],
                                stride,
                                k_chunk,
                                MICRO_M
                            );
                        }
                    }

                    // j方向の端数（右端）も fallback（スカラー）
                    int j_tail = j_max % MICRO_N;
                    if (j_tail > 0) {
                        for (int j = j_max - j_tail; j < j_max; ++j) {
                            for (int ii_inner = 0; ii_inner < MICRO_M; ++ii_inner) {
                                double sum = 0.0;
                                for (int k = kk; k < k_max; ++k) {
                                    sum += A[(i + ii_inner) * stride + k] * B_T[k * stride + j];
                                }
                                C[(i + ii_inner) * stride + j] = sum;
                            }
                        }
                    }
                }
            }
        }
    }

    free(B_T);
}
