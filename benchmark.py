import numpy as np
import random
import time
import csv


def naive_matrix_multiplication(A, B):
    N = len(A)
    result = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] += A[i][k] * B[k][j]
    return result


def generate_binary_matrix_list(N):
    return [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]


def run_benchmarks(naive_max_N=400, numpy_max_N=2000, repeat=1, output_csv="results.csv"):
    sizes = []
    N = 50
    while N <= numpy_max_N:
        sizes.append(N)
        N += 50 if N < naive_max_N else 200

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "NaiveTime", "NumpyTime"])

        for N in sizes:
            print(f"Benchmarking N={N}...")

            A_list = generate_binary_matrix_list(N)
            B_list = generate_binary_matrix_list(N)
            A_np = np.array(A_list, dtype=np.int32)
            B_np = np.array(B_list, dtype=np.int32)

            # NumPy
            total_numpy_time = 0.0
            for _ in range(repeat):
                start = time.perf_counter()
                np.dot(A_np, B_np)
                end = time.perf_counter()
                total_numpy_time += end - start
            numpy_time = total_numpy_time / repeat

            # Naive (if within range)
            if N <= naive_max_N:
                total_naive_time = 0.0
                for _ in range(repeat):
                    start = time.perf_counter()
                    naive_matrix_multiplication(A_list, B_list)
                    end = time.perf_counter()
                    total_naive_time += end - start
                naive_time = total_naive_time / repeat
            else:
                naive_time = ""

            writer.writerow([N, naive_time, numpy_time])

    print(f"✅ 結果を {output_csv} に保存しました")


if __name__ == "__main__":
    run_benchmarks(naive_max_N=400, numpy_max_N=2000, repeat=10)
