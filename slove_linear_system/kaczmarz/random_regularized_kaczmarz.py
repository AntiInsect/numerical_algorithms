import numpy as np
import matplotlib.pyplot as plt
from matrix_generation.mat_with_cond_num import mat_with_cond_num
from tqdm import tqdm 
from collections import deque


def random_regularized_kaczmarz(A,f):
    m = A.shape[0]
    n = A.shape[1]
    x = np.random.rand(n)

    
    res = [np.linalg.norm(A @ x - f)/np.linalg.norm(f)]
    norms = np.array([np.linalg.norm(A[i,:])**2 for i in range(m)])
    
    print("constant regularization")
    N_max = 100000
    np.random.seed(666)
    _lambda = 1 + np.random.rand(N_max)

    for i in tqdm(range(N_max)):
        k = i % m
        x = x + (_lambda[i]*(f[k] - np.dot(A[k,:],x))/norms[k])*A[k,:]
        # update residual
        new_res = np.linalg.norm(A @ x - f)/np.linalg.norm(f)
        res.append(new_res)
        if new_res < 10**(-10):
            break
    return x,res

if __name__ == "__main__":
    row, col = 500, 100
    cond = 200
    matrix = mat_with_cond_num(row, col, 'full', cond)
    
    true_sol = np.random.randn(col)
    b = matrix @ true_sol
    
    x_kacz, res_kacz = random_regularized_kaczmarz(matrix, b)
    print(x_kacz)
    print(np.allclose(x_kacz, true_sol, 10**(-2)))

    # plt.figure(figsize=(14,8))
    # plt.semilogy(res_kacz, label='Random Regularized Kaczmarz')
    # plt.title('Relative residuals', fontsize = 20)
    # plt.ylabel('Value', fontsize = 12)
    # plt.xlabel('Projection number', fontsize = 12)
    # plt.yscale('log')
    # plt.legend()
    # plt.show()

