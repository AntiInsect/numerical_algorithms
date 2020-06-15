import numpy as np


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
    
    return eigenvector.reshape(-1,), mu.item(), iters


if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
         [2.5, 8, 2.5, 2, 1.5],
         [2, 2.5, 9, 2.5, 2],
         [1.5, 2, 2.5, 10, 2.5],
         [1, 1.5, 2, 2.5, 11]]
    matrix = np.matrix(A)

    mu = 17
    
    eigenvector, eigenvalue, iters = rayleigh_quotient_iteration(matrix, mu)

    print("Computed eigenvalue:", eigenvalue, " using ", iters, "iterations")
    print("Computed eigenvector:", eigenvector)
