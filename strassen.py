def add(A, B):
  return [ [a + b for a, b in zip(r1, r2)] for r1, r2 in zip(A, B) ]

def sub(A, B):
  return [ [a - b for a, b in zip(r1, r2)] for r1, r2 in zip(A, B) ]

def split(M):
  m = len(M) // 2
  return [ [row[:m] for row in M[:m]],
           [row[m:] for row in M[:m]],
           [row[:m] for row in M[m:]],
           [row[m:] for row in M[m:]] ]

def join(C11, C12, C21, C22):
  top = [r1 + r2 for r1, r2 in zip(C11, C12)]
  bot = [r1 + r2 for r1, r2 in zip(C21, C22)]
  return top + bot

def strassen(A, B):
  if len(A) == 1: return [[A[0][0] * B[0][0]]]
  A11, A12, A21, A22 = split(A)
  B11, B12, B21, B22 = split(B)
  P1 = strassen(add(A11, A22), add(B11, B22))
  P2 = strassen(add(A21, A22), B11)
  P3 = strassen(A11, sub(B12, B22))
  P4 = strassen(A22, sub(B21, B11))
  P5 = strassen(add(A11, A12), B22)
  P6 = strassen(sub(A21, A11), add(B11, B12))
  P7 = strassen(sub(A12, A22), add(B21, B22))
  C11 = add(sub(add(P1, P4), P5), P7)
  C12 = add(P3, P5)
  C21 = add(P2, P4)
  C22 = add(sub(add(P1, P3), P2), P6)
  return join(C11, C12, C21, C22)