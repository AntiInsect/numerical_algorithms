import numpy as np
from pprint import pprint


def scaled_pivoting(A, b):
    n = len(A)

    P_ = np.identity(n) 

    for k in range(n-1):
        # partial pivoting
        for i in range(k+1, n):
            if abs(A[i, k]) > abs(A[k, k]):   
                A[[k, i]] = A[[i, k]]
                tmp_p = np.identity(n) 
                tmp_p[[k, i]] = tmp_p[[i, k]]
                P_ = tmp_p @ P_

    L = np.asarray(np.identity(n))
    U = np.copy(A)
    P = np.identity(n)
    
    permutation = abs(U[:,0]).argsort()[::-1]
    U = U[permutation] 
    P = P[permutation] @ P_ 

    for k in range(n-1):
        # Forward Elimination
        for j in range(k+1, n):
            q = U[j, k] / U[k, k]             
            L[j, k] = q
            for m in range(k, n):
                U[j, m] -= q * U[k, m]

    L = np.asmatrix(L)
    U = np.asmatrix(np.triu(U))
    P = np.asmatrix(P)
    x = np.zeros(n)

    # Ly = b
    bb = np.asmatrix(P) * np.asmatrix(b)
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        y[i] = bb[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]

    y = np.asmatrix(y).T
    # Ux = y
    x = np.zeros(n)
    Uy = np.asarray(np.hstack((U, y)))

    # Backwards Substitution 
    x[n-1] = Uy[n-1, n] / Uy[n-1, n-1]      
    for i in range (n-1,-1,-1):                 
        z = 0.0                                     
        for j in range(i+1,n):                      
            z += Uy[i, j] * x[j]               
        x[i] = (Uy[i, n] - z) / Uy[i, i]       
    x = np.asmatrix(x).T

    return L, U, P, x


if __name__ == "__main__":
    A = [[4,  1,  2, -3,  5],
         [-3, 3, -1,  4, -2],
         [-1, 2,  5,  1,  3],
         [ 5, 4,  3, -1,  2],
         [ 1, -2, 3, -4,  5]]
    b = [[-16], [20], [-4],[-10],[3]]
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    L, U, P, x = scaled_pivoting(A,b)
    # pprint(L)
    # pprint(U)
    # pprint(P)
    pprint(P.T @ L @ U)
    pprint(x)


    