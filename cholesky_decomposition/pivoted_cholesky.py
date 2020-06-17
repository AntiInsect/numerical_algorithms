import numpy as np
from sklearn.datasets import  make_spd_matrix 
from pprint import pprint


def pivoted_cholesky(matrix, M):
    d = np.diag(matrix).copy()
    N = len(d)
    pi = list(range(N))
    L = np.zeros([M, N])
    err = np.sum(np.abs(d))
    
    m = 0
    while (m < M) and (err > 10**(-6)):
        i = m + np.argmax([d[pi[j]] for j in range(m,N)])
        pi[m], pi[i] = pi[i], pi[m]
        L[m, pi[m]] = np.sqrt(d[pi[m]])

        Apim = matrix[pi[m], :]
        for i in range(m+1, N):
            ip = np.inner(L[:m,pi[m]], L[:m,pi[i]]) if m > 0 else 0
            L[m, pi[i]] = (Apim[pi[i]] - ip) / L[m,pi[m]]
            d[pi[i]] -= pow(L[m,pi[i]],2)
        
        err = np.sum([d[pi[i]] for i in range(m+1,N)])
        m += 1
    return L[:m,:]

if __name__ == "__main__":
    N = 10
    A = make_spd_matrix(N)
    L = pivoted_cholesky(A, M=np.linalg.matrix_rank(A))
    pprint(A)
    pprint(L)
    print(np.allclose(L.T @ L, A, 10**(-6)))
