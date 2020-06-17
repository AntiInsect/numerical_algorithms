import numpy as np


def mat_with_cond_num(row, col, rank, cond):
    
    full_rank = False
    while not full_rank:
        sample_mx = np.random.rand(row, col)
        U, sing_vals, V = np.linalg.svd(sample_mx)
        if np.all(~np.isclose(sing_vals, 0.0)):
            full_rank = True

    if rank == 'full':
        rank = min(row, col)

    sing_vals = np.linspace(1.0, cond, rank)[::-1]

    if rank < min(row, col):
        sing_vals = np.concatenate([sing_vals, np.zeros(min(row, col) - rank)])

    Sigma = np.diag(sing_vals)
    if Sigma.shape[0] < row:
        Sigma = np.vstack([Sigma, np.zeros((row - Sigma.shape[0], col))])
    if Sigma.shape[1] < col:
        Sigma = np.hstack([Sigma, np.zeros((row, col - Sigma.shape[1]))])

    return U @ Sigma @ V

if __name__ == "__main__":
    row, col, rank, cond= 7, 6, 5, 4
    matrix = mat_with_cond_num(row, col, rank, cond)
    print(matrix)

