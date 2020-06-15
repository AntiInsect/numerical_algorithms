import numpy as np
from pprint import pprint

# Orthogonal Iteration is a block version of the Power Method,
# also sometimes called "Simultaneous (Power) Interation"
def orthogonal_iteration(matrix, k):
    n, m = matrix.shape
    
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
 
    iters = 0
    converge = False

    while not converge:
        iters += 1

        Z = matrix.dot(Q)
        Q, R = np.linalg.qr(Z)

        # can use other stopping criteria as well 
        converge = (np.square(Q - Q_prev)).sum() <= 1e-3
        Q_prev = Q

    return np.diag(R), Q, iters

def qr_algorithm(matrix):
    return orthogonal_iteration(matrix, matrix.shape[0]) 


if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
        [2.5, 8, 2.5, 2, 1.5],
        [2, 2.5, 9, 2.5, 2],
        [1.5, 2, 2.5, 10, 2.5],
        [1, 1.5, 2, 2.5, 11]]

    eps = 10**(-10)

    matrix = np.matrix(A)
    k = 3
    R, Q, iters = orthogonal_iteration(matrix, k)  

    pprint(R)
    pprint(Q)
    pprint(iters)

    R, Q, iters = qr_algorithm(matrix)

    pprint(R)
    pprint(Q)
    pprint(iters)
