import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *



# Alternating least squares
def alternating_least_squares(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Parameters:
    -----------
    A : m x n array matrix to complete
    mask : m x n array matrix with entries zero (if missing) or one (if present)
    k : integer how many factors to use
    mu : float hyper-parameter penalizing norm of factored U, V
    epsilon : float convergence condition on the difference between iterative results
    max_iterations: int hard limit on maximum number of iterations

    Returns:
    --------
    X: m x n array completed matrix 
    """

    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)

    for _ in tqdm(range(max_iterations)):

        # iteration for U
        for i in range(m):
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))

        # iteration for V
        for j in range(n):
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))

        X = np.dot(U, V.T)

        if np.linalg.norm(X - prev_X) / m / n < epsilon:
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

    R_hat = alternating_least_squares(R, mask, k, mu)
    print("RMSE:", get_rmse(U, V, R_hat, mask))

    plt.imshow(R_hat.T)
    plt.title("The the matrix after completion")
    plt.show()
