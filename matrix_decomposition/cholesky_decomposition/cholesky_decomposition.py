from math import sqrt
from pprint import pprint
from sklearn.datasets import  make_spd_matrix 
import numpy as np


def cholesky_decomposition(A):    
    n = len(A)
    
    L = [[0.0] * n for i in range(n)]
    
    for i in range(n):
        for k in range(i+1):
            suma = sum(L[i][j]* L[k][j] for j in range(k))
            if (i==k):
                L[i][k] = sqrt(A[i][i] - suma)
            else:
                L[i][k] = (1.0 / L[k][k]* (A[i][k] - suma))
    return np.array(L)

if __name__ == "__main__":
    # Sample symmetric, postive matrix
    order = 10
    A = make_spd_matrix(order)
    L = cholesky_decomposition(A)
    pprint(A)
    pprint(L)
    pprint(np.allclose(A, L @ L.T, 10**(-10)))
