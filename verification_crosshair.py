from typing import List

def spec_matmul(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    l = len(A)
    m = len(A[0])
    n = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(n)] for i in range(l)]

def matrix_mul_verified(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """
    pre: len(A) > 0 and len(B) > 0
    pre: all(len(row) == len(A[0]) for row in A)
    pre: all(len(row) == len(B[0]) for row in B)
    pre: len(A[0]) == len(B)
    post: _ == spec_matmul(A, B)
    """
    l = len(A)
    m = len(A[0])
    n = len(B[0])
    C = [[0 for _ in range(n)] for _ in range(l)]
    for i in range(l):
        for j in range(n):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C