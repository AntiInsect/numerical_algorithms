from math import copysign
import numpy as np
from pprint import pprint


def householder_reflection(matrix):

    num_rows, num_cols = np.shape(matrix)
    Q = np.identity(num_rows)
    R = np.copy(matrix)

    # Iterative over column sub-vector and
    # compute Householder matrix to zero-out
    # lower triangular matrix entries.
    for i in range(num_rows - 1):
        x = R[i:, i]

        e = np.zeros_like(x)
        e[0] = copysign(np.linalg.norm(x), -matrix[i, i])
        u = x + e
        v = u / np.linalg.norm(u)

        Q_cnt = np.identity(num_rows)
        Q_cnt[i:, i:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_cnt, R)
        Q = np.dot(Q, Q_cnt.T)

    return (Q, R)


if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
        [2.5, 8, 2.5, 2, 1.5],
        [2, 2.5, 9, 2.5, 2],
        [1.5, 2, 2.5, 10, 2.5],
        [1, 1.5, 2, 2.5, 11]]

    eps = 10**(-10)

    matrix = np.matrix(A)
    Q , R = householder_reflection(matrix)  
    
    pprint(Q)
    pprint(R)
    assert np.linalg.norm(Q @ R - A) <= eps
