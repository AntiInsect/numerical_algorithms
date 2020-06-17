import numpy as np
import matplotlib.pyplot as plt
from matrix_generation.mat_with_cond_num import mat_with_cond_num
from tqdm import tqdm 


def classic_kaczmarz(A, f):
    
    m = A.shape[0]
    n = A.shape[1]
    x = np.random.rand(n)
    
    # calculate row norms in advance    
    norms = np.array([np.linalg.norm(A[i,:])**2 for i in range(m)])
    
    # Kaczmarz
    res = np.array([np.linalg.norm(A @ x - f)/np.linalg.norm(f)])
    for i in tqdm(range(200000)):
        k = i % m
        x = x + ((f[k] - np.dot(A[k,:],x))/norms[k])*A[k,:]

        # update residual
        new_res = np.linalg.norm(A @ x - f)/np.linalg.norm(f)
        res = np.append(res,[new_res])
        if new_res < 10**(-5):
            break

    return x, res


if __name__ == "__main__":
    row, col = 500, 100
    cond = 200
    matrix = mat_with_cond_num(row, col, 'full', cond)
    
    true_sol = np.random.randn(col)
    b = matrix @ true_sol
    
    x_kacz, res_kacz = classic_kaczmarz(matrix, b)
    print(x_kacz)
    print(np.allclose(x_kacz, true_sol, 10**(-2)))

    # plt.figure(figsize=(14,8))
    # plt.semilogy(res_kacz, label='Kaczmarz')
    # plt.title('Relative residuals', fontsize = 20)
    # plt.ylabel('Value', fontsize = 12)
    # plt.xlabel('Projection number', fontsize = 12)
    # plt.yscale('log')
    # plt.legend()
    # plt.show()

