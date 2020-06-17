import numpy as np
from tqdm import tqdm
from sklearn.datasets import  make_spd_matrix 
from pprint import pprint



def gauss_seidel(A, b):
    N = A.shape[0]
    x = np.zeros(N)
    for i in range(1000000):
        x_prev = np.array(x)

        for j in range(N):
            x[j] = (b[j] - np.sum([A[j][k] * x[k] for k in range(N) if k != j])) / A[j][j]

        norm = np.sum(np.abs(x - x_prev)) / (np.sum(np.abs(x_prev)) + 10**(-5))

        if norm < 10**(-14):
            return x, i
    print("Doesn't converge.")
    return None, -1


if __name__ == "__main__":
    N = 10
    A = make_spd_matrix(N)
    b = np.random.randint(N, size=N)
    x, iters = gauss_seidel(A, b)
    print(x)
    print(iters)
