import numpy as np
import matplotlib.pyplot as plt
from matrix_generation.band_mat_with_cond_num import band_mat_with_cond_num
from tqdm import tqdm 
from collections import deque


def random_lyusternick_kaczmarz(A,b):    
    n = len(A)
    A = A.astype(float)
    b = b.astype(float)
    
    # Norm each equation
    normCoef = np.sqrt(np.sum(A ** 2, 1)).T
    A /= normCoef[:, None]
    b /= normCoef
    x = np.random.rand(n)

    # using 2nd norm below (change to suit your needs)
    disparity = lambda x: np.linalg.norm(b - np.dot(A, np.transpose(x)))/np.linalg.norm(b)
    cosVectAngle = lambda a, b: np.inner(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    resudial = lambda x: np.linalg.norm(A @ x - b) / np.linalg.norm(b)

    HPlane_Pnts = deque()
    lastVr = None
    q_last = None
    init_lastVr = False

    res = [disparity(x)]
    for _ in tqdm(range(200000)):

        if disparity(x) < 10**(-5):
            break

        for i in range(n):
            
            # Randomized here
            k = np.random.randint(n)
            t = A[k] @ x - b[k]
            x -= A[k] * t

            res.append(resudial(x))

            if i == n - 1:
                HPlane_Pnts.append(np.copy(x))
        
        if len(HPlane_Pnts) > 3:
            HPlane_Pnts.popleft()

            prevVr = HPlane_Pnts[-2] - HPlane_Pnts[-3] if not init_lastVr else lastVr
            lastVr = HPlane_Pnts[-1] - HPlane_Pnts[-2]

            init_lastVr = True
            q_prev = q_last
            q_last = np.linalg.norm(lastVr) / np.linalg.norm(prevVr)

            if q_prev is not None and q_last != 1 and \
               np.isclose(q_prev, q_last) and np.isclose(cosVectAngle(prevVr, lastVr), 1):
               # Lyusternik acceleration
               x = HPlane_Pnts[-1] + (HPlane_Pnts[-1] - HPlane_Pnts[-2]) / (1.0 - q_last)
               HPlane_Pnts.clear() 

    res.append(resudial(x))
    return x, res


if __name__ == "__main__":
    row, col = 6, 6
    half_band_size, ampl, cond = 3, 50, 50
    matrix = band_mat_with_cond_num(row, col, half_band_size, ampl, cond)
    
    true_sol = np.random.randn(row)
    b = matrix @ true_sol
    
    x_kacz, res_kacz = random_lyusternick_kaczmarz(matrix, b)
    print(x_kacz)
    print(np.allclose(x_kacz, true_sol, 10**(-2)))

    # plt.figure(figsize=(14,8))
    # plt.semilogy(res_kacz, label='Lyusternick Kaczmarz')
    # plt.title('Relative residuals', fontsize = 20)
    # plt.ylabel('Value', fontsize = 12)
    # plt.xlabel('Projection number', fontsize = 12)
    # plt.yscale('log')
    # plt.legend()
    # plt.show()