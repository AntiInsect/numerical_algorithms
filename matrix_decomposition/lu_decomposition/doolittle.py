import numpy as np
from pprint import pprint


def doolittle(A):
	n = A.shape[0]
	P = np.identity(n)
	permutation = abs(A[:,0]).argsort()[::-1]
	A = A[permutation]
	P = P[permutation]

	L = np.identity(n)
	U = np.zeros((n,n))
	for i in range(n):
		for k in range(n):
			if i <= k:
				U[i,k] = A[i,k] - sum([L[i,j]*U[j,k] for j in range(0,i)])
			else:
				L[i,k] = (A[i,k] - sum([L[i,j]*U[j,k] for j in range(0,i)])) / U[k,k]
	
	return L, U, P


if __name__ == "__main__":
    A = [[4,  1,  2, -3,  5],
         [-3, 3, -1,  4, -2],
         [-1, 2,  5,  1,  3],
         [ 5, 4,  3, -1,  2],
         [ 1, -2, 3, -4,  5]]

    A = np.asarray(A, dtype=np.float64)

    L,U,P = doolittle(A)

    pprint(L)
    pprint(U)
    pprint(P)
    pprint(P.T @ L @ U)





