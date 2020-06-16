import numpy as np


def power_iteration(matrix, approx, eps):
    
    # we do with column vector
    eigenvector = approx.T / np.linalg.norm(approx)
    iters = 0
    converge = False

    while not converge:
        iters += 1
        eigenvector_new = matrix @ eigenvector
        
        # Normalization
        eigenvector_new /= np.linalg.norm(eigenvector_new)

        converge = np.linalg.norm(eigenvector_new - eigenvector) <= eps
        eigenvector = eigenvector_new

    # compute the Rayleigh quotient
    eigenvalue_max = np.dot(eigenvector.T, matrix * eigenvector) / np.dot(eigenvector.T, eigenvector)
    
    return eigenvector.reshape(-1,), eigenvalue_max.item(), iters


if __name__ == "__main__":
    eps = 10**(-11)
    
    A = [[7, 2.5, 2, 1.5, 1],
         [2.5, 8, 2.5, 2, 1.5],
         [2, 2.5, 9, 2.5, 2],
         [1.5, 2, 2.5, 10, 2.5],
         [1, 1.5, 2, 2.5, 11]]
    b = [np.random.rand() for i in range(5)]

    matrix = np.matrix(A)
    vector = np.matrix(b)

    eigenvector, eigenvalue_max, iters = power_iteration(matrix, vector, eps)
    print("Computed eigenvalue:", eigenvalue_max, " using ", iters, "iterations")
    print("Computed eigenvector:", eigenvector)
