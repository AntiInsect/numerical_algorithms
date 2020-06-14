import numpy as np
from pprint import pprint


def gram_schmidt_process(matrix):
    (num_rows, num_cols) = np.shape(matrix)

    # Initialize empty orthogonal matrix Q.
    Q = np.empty([num_rows, num_rows])
    cnt = 0

    # Compute orthogonal matrix Q.
    for a in matrix.T:
        u = np.copy(a)
        for i in range(cnt):
            proj = np.dot(np.dot(Q[:, i].T.reshape(-1, 1), a), Q[:, i])
            u -= proj

        e = u / np.linalg.norm(u)
        Q[:, cnt] = e

        cnt += 1  # Increase columns counter.

    # Compute upper triangular matrix R.
    R = np.dot(Q.T, matrix)

    return (Q, R)


if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
        [2.5, 8, 2.5, 2, 1.5],
        [2, 2.5, 9, 2.5, 2],
        [1.5, 2, 2.5, 10, 2.5],
        [1, 1.5, 2, 2.5, 11]]

    eps = 10**(-10)

    matrix = np.matrix(A)
    Q , R = gram_schmidt_process(matrix)  
    
    pprint(Q)
    pprint(R)
    assert np.linalg.norm(Q @ R - A) <= eps
