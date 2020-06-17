from math import hypot
import numpy as np
from pprint import pprint


def givens_rotations(matrix):
    num_rows, num_cols = np.shape(matrix)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(matrix)

    # Iterate over lower triangular matrix.
    rows, cols = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:

            # Compute matrix entries for Givens rotation
            r = hypot(R[col, col], R[row, col])
            c = R[col, col] / r
            s = - R[row, col] / r

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)

if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
        [2.5, 8, 2.5, 2, 1.5],
        [2, 2.5, 9, 2.5, 2],
        [1.5, 2, 2.5, 10, 2.5],
        [1, 1.5, 2, 2.5, 11]]

    eps = 10**(-10)

    matrix = np.matrix(A)
    Q , R = givens_rotations(matrix)  
    
    pprint(Q)
    pprint(R)
    assert np.linalg.norm(Q @ R - A) <= eps
