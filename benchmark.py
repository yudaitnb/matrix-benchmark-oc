import numpy as np
import random
import time
import csv
import ctypes
import os
from typing import Callable, List, Optional
from collections import Counter

# =======================
# Utility Functions
# =======================

def assert_aligned(np_array, alignment=32, name="array"):
    address = np_array.__array_interface__['data'][0]
    if address % alignment != 0:
        raise RuntimeError(f"{name} is not aligned to {alignment} bytes: address=0x{address:x}")

def aligned_array(shape, dtype=np.float64, alignment=32):
    n_bytes = np.prod(shape) * np.dtype(dtype).itemsize
    buf = np.zeros(n_bytes + alignment, dtype=np.uint8)
    start_index = -buf.ctypes.data % alignment
    aligned_buf = buf[start_index:start_index + n_bytes].view(dtype)
    return aligned_buf.reshape(shape)

def generate_binary_matrix_numpy(N, dtype=np.float64, alignment=32):
    stride = ((N * dtype().itemsize + alignment - 1) // alignment) * alignment // dtype().itemsize
    mat = aligned_array((N, stride), dtype=dtype, alignment=alignment)
    mat[:, :N] = np.random.randint(0, 2, size=(N, N))
    return mat, stride  # strideを一緒に返す

def load_ctypes_library(filename: str, func_name: str, argtypes: List):
    so_path = os.path.join(os.path.dirname(__file__), "build", filename)
    lib = ctypes.cdll.LoadLibrary(so_path)
    func = getattr(lib, func_name)
    func.argtypes = argtypes
    return func

# =======================
# Benchmark Implementations
# =======================

class BenchmarkImplementation:
    def __init__(self, name: str, enabled: Callable[[int], bool]):
        self.name = name
        self.enabled = enabled

    def run(self, A_list, B_list, A_np, B_np) -> float:
        """
        行列積の処理時間（秒）を返す。
        実行内容の時間計測は各サブクラスで行うこと。
        """
        raise NotImplementedError("Subclasses must implement this method")


class NaivePython(BenchmarkImplementation):
    def __init__(self, max_N):
        super().__init__("Naive", lambda N: N <= max_N)

    def run(self, A_list, B_list, A_np, B_np) -> float:
        N = len(A_list)
        result = [[0 for _ in range(N)] for _ in range(N)]

        start = time.perf_counter()
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    result[i][j] += A_list[i][k] * B_list[k][j]
        end = time.perf_counter()

        return end - start

class NaiveC(BenchmarkImplementation):
    def __init__(self, max_N):
        super().__init__("NaiveC", lambda N: N <= max_N)
        self.func = load_ctypes_library("matmul_naive.so", "matmul_naive_c", [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ])

    def run(self, A_list, B_list, A_np, B_np) -> float:
        N = A_np.shape[0]
        C = np.zeros((N, N), dtype=np.float64)

        start = time.perf_counter()
        self.func(A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N)
        end = time.perf_counter()

        return end - start

class BlockedC(BenchmarkImplementation):
    def __init__(self, max_N):
        super().__init__("BlockedC", lambda N: N <= max_N)
        self.func = load_ctypes_library("matmul_blocked.so", "matmul_blocked_c", [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int
        ])

    def run(self, A_list, B_list, A_np, B_np) -> float:
        N = A_np.shape[0]
        C = np.zeros((N, N), dtype=np.float64)

        start = time.perf_counter()
        self.func(A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N, 64)
        end = time.perf_counter()

        return end - start

class BlockedOMP(BenchmarkImplementation):
    def __init__(self):
        super().__init__("BlockedOMP", lambda N: True)
        self.func = load_ctypes_library("matmul_blocked_omp.so", "matmul_blocked_omp_c", [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int
        ])

    def run(self, A_list, B_list, A_np, B_np) -> float:
        N = A_np.shape[0]
        C = np.zeros((N, N), dtype=np.float64)

        start = time.perf_counter()
        self.func(A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N, 64)
        end = time.perf_counter()

        return end - start
    
class BlockedOMPTuning(BenchmarkImplementation):
    def __init__(self):
        super().__init__("BlockedOMPTuning", lambda N: True)
        self.func = load_ctypes_library("matmul_blocked_omp_tuning.so", "matmul_blocked_omp_tuning_c", [
            ctypes.POINTER(ctypes.c_double),  # A
            ctypes.POINTER(ctypes.c_double),  # B
            ctypes.POINTER(ctypes.c_double),  # C
            ctypes.c_int,  # N
            ctypes.c_int,  # stride
            ctypes.c_int   # block_size
        ])
        self.best_block_sizes = {}

    def autotune_block_size(self, N, stride, repeat=5, candidates=None):
        if N in self.best_block_sizes:
            return self.best_block_sizes[N]

        if candidates is None:
            candidates = [16, 24, 32, 48, 64, 96, 128, 160, 192, 256]

        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)
        A_aligned = aligned_array((N, stride), dtype=np.float64, alignment=32)
        B_aligned = aligned_array((N, stride), dtype=np.float64, alignment=32)
        np.copyto(A_aligned[:, :N], A)
        np.copyto(B_aligned[:, :N], B)

        assert_aligned(A_aligned, 32, "A")
        assert_aligned(B_aligned, 32, "B")

        results = []

        for _ in range(repeat):
            best_time = float("inf")
            best_block = candidates[0]

            for block_size in candidates:
                C = aligned_array((N, stride), dtype=np.float64, alignment=32)
                A_ptr = A_aligned.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                B_ptr = B_aligned.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                start = time.perf_counter()
                self.func(A_ptr, B_ptr, C_ptr, N, stride, block_size)
                end = time.perf_counter()
                elapsed = end - start

                if elapsed < best_time:
                    best_time = elapsed
                    best_block = block_size

            results.append(best_block)

        # 最頻値を選ぶ
        counter = Counter(results)
        most_common_block, count = counter.most_common(1)[0]
        self.best_block_sizes[N] = most_common_block
        print(f"Autotuned best block size for N={N} is {most_common_block} (votes: {dict(counter)})")
        return most_common_block

    def run(self, A_list, B_list, A_np, B_np) -> float:
        N = A_np.shape[0]
        A_aligned, stride = generate_binary_matrix_numpy(N, alignment=32)
        B_aligned, _ = generate_binary_matrix_numpy(N, alignment=32)
        C = aligned_array((N, stride), dtype=np.float64, alignment=32)

        assert_aligned(A_aligned, 32, "A_np")
        assert_aligned(B_aligned, 32, "B_np")
        assert_aligned(C, 32, "C")

        block_size = self.autotune_block_size(N, stride)
        A_ptr = A_aligned.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        B_ptr = B_aligned.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        start = time.perf_counter()
        self.func(A_ptr, B_ptr, C_ptr, N, stride, block_size)
        end = time.perf_counter()
        return end - start



class NumpyDot(BenchmarkImplementation):
    def __init__(self, max_N):
        super().__init__("Numpy", lambda N: N <= max_N)

    def run(self, A_list, B_list, A_np, B_np) -> float:
        start = time.perf_counter()
        np.dot(A_np, B_np)
        end = time.perf_counter()
        return end - start

# =======================
# Main Benchmark Loop
# =======================

def run_benchmarks(naive_max_N=400, naive_c_max_N=1600, sequential_c_max_N=2000, numpy_max_N=4000, repeat=1, output_csv="results.csv"):
    sizes = []
    N = 50
    while N <= numpy_max_N:
        sizes.append(N)
        N += 50 if N < naive_max_N else 200

    implementations = [
        NaivePython(naive_max_N),
        NaiveC(naive_c_max_N),
        BlockedC(sequential_c_max_N),
        BlockedOMP(),
        BlockedOMPTuning(),
        NumpyDot(numpy_max_N),
    ]

    # ✅ BlockedOMPTuning のブロックサイズを事前に推定
    print("🔧 事前に BlockedOMPTuning のブロックサイズをチューニング...")
    for impl in implementations:
        if isinstance(impl, BlockedOMPTuning):
            for N in sizes:
                if impl.enabled(N):
                    _, stride = generate_binary_matrix_numpy(N, dtype=np.float64)
                    impl.autotune_block_size(N, stride, repeat=repeat)

    with open(os.path.join("result", output_csv), mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N"] + [impl.name for impl in implementations])

        for N in sizes:
            print(f"\n🚀 Benchmarking N={N}...")
            A_np, stride = generate_binary_matrix_numpy(N, dtype=np.float64)
            B_np, _ = generate_binary_matrix_numpy(N, dtype=np.float64)
            A_list = A_np[:, :N].tolist()
            B_list = B_np[:, :N].tolist()

            row = [N]
            for impl in implementations:
                if impl.enabled(N):
                    times = []
                    for _ in range(repeat):
                        if isinstance(impl, BlockedOMPTuning):
                            elapsed = impl.run(A_list, B_list, A_np, B_np)
                        else:
                            elapsed = impl.run(A_list, B_list, A_np[:, :N], B_np[:, :N])
                        times.append(elapsed)
                    avg_time = sum(times) / len(times)
                    row.append(avg_time)
                    print(f"{impl.name}: {avg_time:.6f} sec")
                else:
                    row.append("")
            writer.writerow(row)

    print(f"\n✅ 結果を {output_csv} に保存しました")


if __name__ == "__main__":
    run_benchmarks(naive_max_N=400, naive_c_max_N=1600, sequential_c_max_N=2000, numpy_max_N=4000, repeat=1)