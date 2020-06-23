import numpy as np
import matplotlib.pyplot as plt
from matrix_generation.mat_with_cond_num import mat_with_cond_num
from tqdm import tqdm 
from collections import deque
import scipy


def extended_kaczmarz(A, b_):
    if len(b_.shape) == 1:
        b = b_[:, np.newaxis].copy()
    else:
        b = b_.copy()
    z = b.copy()
    m, n = A.shape
    x = np.random.rand(n)

    res = [np.linalg.norm(A @ x - b)/np.linalg.norm(b)]
    for _ in tqdm(range(100000)):
        
        col_idx = np.random.choice(np.arange(n), 1, replace=False)
        row_idx = np.random.choice(np.arange(m), 1, replace=False)
        
        x = x + (b[row_idx] - z[row_idx] - A[row_idx, :] @ x) / np.sum(A[row_idx, :]**2) * A[row_idx, :].T
        z -= (z.T @ A[:, col_idx]) / np.sum(A[:, col_idx]**2) * A[:, col_idx]

        new_res = np.linalg.norm(A @ x - b)/np.linalg.norm(b)
        res.append(new_res)
        if new_res < 10**(-10):
            break

    return x, res

if __name__ == "__main__":
    row, col = 500, 100
    cond = 200
    matrix = mat_with_cond_num(row, col, 'full', cond)
    
    true_sol = np.random.randn(col)
    b = matrix @ true_sol
    
    x_kacz, res_kacz  = extended_kaczmarz(matrix, b)
    print(x_kacz)
    print(np.allclose(x_kacz, true_sol, 10**(-2)))

    # plt.figure(figsize=(14,8))
    # plt.semilogy(res_kacz, label='Extended Kaczmarz')
    # plt.title('Relative residuals', fontsize = 20)
    # plt.ylabel('Value', fontsize = 12)
    # plt.xlabel('Projection number', fontsize = 12)
    # plt.yscale('log')
    # plt.legend()
    # plt.show()