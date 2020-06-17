import numpy as np
from tqdm import tqdm
from sklearn.datasets import  make_spd_matrix 
from pprint import pprint



def gauss_seidel(A, b, x, N):
    for i in range(1000000):
        x_prev = np.array(x)

        for j in range(N):
            x[j] = (b[j] - np.sum([A[j][k] * x[k] for k in range(N) if k != j])) / A[j][j]

        norm = np.sum(np.abs(x - x_prev)) / (np.sum(np.abs(x_prev)) + 10**(-5))

        if norm < 10**(-14):
            print("Sequence converges to [", end="")
            for j in range(N - 1):
                print(x[j], ",", end="")
            print(x[N - 1], "]. Took", i + 1, "iterations.")
            return
    print("Doesn't converge.")

if __name__ == "__main__":
    # matrix2 = [[3.0, 1.0], [2.0, 6.0]]
    # vector2 = [5.0, 9.0]
    # guess = [0.0, 0.0]

    N = 10
    A = make_spd_matrix(N)
    b = np.random.randint(N, size=N)
    x = np.zeros(N)
    pprint(b)
    pprint(x)
    gauss_seidel(A, b, x, N)
