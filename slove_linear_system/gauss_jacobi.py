import numpy as np
from sklearn.datasets import  make_spd_matrix 


def gauss_jacobi(A,b):
    n = A.shape[0]
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    D = np.zeros((n,n))
    
    x0 = np.random.rand(n)
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i,j] = A[i,j]
            elif i==j:
                D[i,j] = A[i,j]
            else:
                U[i,j] = A[i,j]
    x1 = x0
    x0 = np.reshape(np.zeros((n,1)),(n,1))
    
    del_x = x1 - x0
    iters = 0
    while np.linalg.norm(del_x) > 10**(-5):
        iters += 1
        
        x0 = x1
        x1 = -1 * np.linalg.inv(D) @ (L+U) @ x0 + np.linalg.inv(D) @ b
        
        del_x = x1 - x0
    
    return x1, iters

if __name__ == "__main__":

    N = 5
    A = make_spd_matrix(N)
    b = np.random.randint(N, size=N)
    x, iters = gauss_jacobi(A,b)
    print(x)
    print(iters)
    