import numpy as np
import matplotlib.pyplot as plt
from matrix_generation.band_mat_with_cond_num import band_mat_with_cond_num
from tqdm import tqdm 


def random_kaczmarz(A,f):
    m = A.shape[0]
    x = np.random.rand(m)

    # calculate row norms in advance
    norms = np.array([np.linalg.norm(A[i,:])**2 for i in range(m)])
    cum_norms = np.cumsum(norms)
    cum_norms /= cum_norms[len(cum_norms)-1]
    
    np.random.seed(666)
    res = np.array([np.linalg.norm(A @ x - f)/np.linalg.norm(f)])
    for i in tqdm(range(200000)):
        r = np.random.rand()
        k = np.searchsorted(cum_norms,r)
        x = x + ((f[k] - np.dot(A[k,:],x))/norms[k])*A[k,:]
        #update residual
        new_res = np.linalg.norm(A @ x - f)/np.linalg.norm(f)
        res = np.append(res,[new_res])
        if new_res < 10**(-5):
            break
    return x, res

if __name__ == "__main__":
    row, col = 6, 6
    half_band_size, ampl, cond = 3, 50, 50
    matrix = band_mat_with_cond_num(row, col, half_band_size, ampl, cond)
    
    true_sol = np.random.randn(row)
    b = matrix @ true_sol
    
    x_kacz, res_kacz = random_kaczmarz(matrix, b)
    print(x_kacz)
    print(np.allclose(x_kacz, true_sol, 10**(-2)))

    # plt.figure(figsize=(14,8))
    # plt.semilogy(res_kacz, label='Random Kaczmarz')
    # plt.title('Relative residuals', fontsize = 20)
    # plt.ylabel('Value', fontsize = 12)
    # plt.xlabel('Projection number', fontsize = 12)
    # plt.yscale('log')
    # plt.legend()
    # plt.show()