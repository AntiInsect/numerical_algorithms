import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *


# Biased alternating least squares
def biased_alternating_least_squares(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Parameters:
    -----------
    A : m x n array matrix to complete
    mask : m x n array matrix with entries zero (if missing) or one (if present)
    k : integer how many factors to use
    mu : float hyper-parameter penalizing norm of factored U, V and biases beta, gamma
    epsilon : float convergence condition on the difference between iterative results
    max_iterations: int hard limit on maximum number of iterations

    Returns:
    --------
    X: m x n array completed matrix
    """

    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    beta = np.random.randn(m)
    gamma = np.random.randn(n)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T) + \
             np.outer(beta, np.ones(n)) + np.outer(np.ones(m), gamma)

    for _ in tqdm(range(max_iterations)):

        A_tilde = A - np.outer(np.ones(m), gamma)
        V_tilde = np.c_[np.ones(n), V]
        # iteration for U

        for i in range(m):
            U_tilde = np.linalg.solve(np.linalg.multi_dot([V_tilde.T, C_u[i], V_tilde]) +
                                      mu * np.eye(k + 1),
                                      np.linalg.multi_dot([V_tilde.T, C_u[i], A_tilde[i,:]]))
            beta[i] = U_tilde[0]
            U[i] = U_tilde[1:]

        A_tilde = A - np.outer(beta, np.ones(n))
        U_tilde = np.c_[np.ones(m), U]

        # iteration for V
        for j in range(n):
            V_tilde = np.linalg.solve(np.linalg.multi_dot([U_tilde.T, C_v[j], U_tilde]) +
                                      mu * np.eye(k + 1),
                                      np.linalg.multi_dot([U_tilde.T, C_v[j], A_tilde[:,j]]))
            gamma[j] = V_tilde[0]
            V[j] = V_tilde[1:]

        X = np.dot(U, V.T) + \
            np.outer(beta, np.ones(n)) + np.outer(np.ones(m), gamma)

        if  np.linalg.norm(X - prev_X) / m / n < epsilon:
            break
        
        prev_X = X

    return X


if __name__ == "__main__":
    m, n = 200, 160
    k = 10
    noise = 0.1
    mask_prob = 0.75
    mu = 1e-2

    U, V, R = gen_factorization_with_noise(m, n, k, noise)
    mask = gen_mask(m, n, mask_prob)

    plt.imshow(R.T)
    plt.title("The matrix to be complete")
    plt.show()

    plt.imshow(mask.T)
    plt.title("The the mask")
    plt.show()

    R_hat = biased_alternating_least_squares(R, mask, k, mu)
    print("RMSE:", get_rmse(U, V, R_hat, mask))

    plt.imshow(R_hat.T)
    plt.title("The the matrix after completion")
    plt.show()
