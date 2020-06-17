import numpy as np
from tqdm import tqdm


# Method finding all eigenpairs of a symmetric matrix
def qr_algorithm(A):
    backup = np.array(A)
    
    A = hessenberg_decomposition(A)
    M = A.shape[0]
    
    for k in range(1000):
        # set the shift to be the last diagonal element
        mu = A[M-1, M-1] 
        Q, R = np.linalg.qr(A - mu * np.eye(M))
        A_new = R @ Q + mu * np.eye(M)
        if np.all(np.abs(A_new - A) < 1e-8):
            break
        A = A_new
        
        mus = np.diag(A)
        eigvecs = np.zeros_like(A)
        eigvals = np.zeros(A.shape[0])
        
        for i, mu in enumerate(mus):
            eigvec, eigval, _ = rayleigh_quotient_iteration(backup, mu)
            eigvecs[:, i] = eigvec
            eigvals[i] = eigval
        
        return eigvals, eigvecs

# Reduce a square matrix to upper Hessenberg form using Householder reflections.
def hessenberg_decomposition(A):
    A = np.array(A)
    M, _ = A.shape
    vs = []
    for i in range(M-2):
        a = A[i+1:, i]
        c = np.linalg.norm(a)
        s = np.sign(a[0])
        e = np.zeros(len(a))
        v = a + s*c*e
        vs.append(v)
        
        # left transform
        for j in range(i, M):
            A[i+1:, j] = A[i+1:, j] - (2 * v.T @ A[i+1:, j]) / (v.T @ v) * v
        # right transform
        for j in range(i, M):
            A[j, i+1:M] = A[j, i+1:M] - 2 * ((A[j, i+1:M].T @ v) / (v.T @ v)) * v.T
        
    return A


def rayleigh_quotient_iteration(matrix, mu):
    
    # we do with column vector
    approx = np.matrix([np.random.rand() for i in range(5)])
    eigenvector = approx.T / np.linalg.norm(approx)

    matrix_new = np.linalg.inv(matrix - mu * np.eye(matrix.shape[0]))

    iters = 0
    converge = False

    while not converge:
        iters += 1
        eigenvector_new = matrix_new @ eigenvector
        
        # Normalization
        eigenvector_new /= np.linalg.norm(eigenvector_new)
        converge = np.linalg.norm(eigenvector_new - eigenvector) <= 10**(-11)
        eigenvector = eigenvector_new

        # compute the Rayleigh quotient
        mu = np.dot(eigenvector.T, matrix * eigenvector) / np.dot(eigenvector.T, eigenvector)

        if iters > 10:
            break
    
    return eigenvector.reshape(-1,), mu.item(), iters

if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
        [2.5, 8, 2.5, 2, 1.5],
        [2, 2.5, 9, 2.5, 2],
        [1.5, 2, 2.5, 10, 2.5],
        [1, 1.5, 2, 2.5, 11]]
    matrix = np.matrix(A)
    eigvals, eigvecs = qr_algorithm(A)
    print(eigvals)
    print(eigvecs)
