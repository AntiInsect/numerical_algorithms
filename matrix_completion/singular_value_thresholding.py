import numpy as np
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds
from utils import *
import matplotlib.pyplot as plt



# Solve using iterative singular value thresholding.
def singular_value_thresholding(A, mask, tau=None, delta=None, epsilon=1e-2, max_iterations=500, algorithm='randomized'):
    """    
    Parameters:
    -----------
    A : m x n array, matrix to complete
    mask : m x n array, matrix with entries zero (if missing) or one (if present)
    tau : float singular value thresholding amount;, default to 5 * (m + n) / 2
    delta : float step size per iteration; default to 1.2 times the undersampling ratio
    epsilon : float convergence condition on the relative reconstruction error
    max_iterations: int hard limit on maximum number of iterations

    algorithm: str, 'arpack' or 'randomized'
        SVD solver to use. Either 'arpack' for the ARPACK wrapper in 
        SciPy (scipy.sparse.linalg.svds), or 'randomized' for the 
        randomized algorithm due to Halko (2009).

    Returns:
    --------
    X: m x n array completed matrix
    """

    if algorithm not in ['randomized', 'arpack']:
        raise ValueError("unknown algorithm %r" % algorithm)
    
    Y = np.zeros_like(A)
    if tau is None:
        tau = 5 * np.sum(A.shape) / 2
    if delta is None:
        delta = 1.2 * np.prod(A.shape) / np.sum(mask)

    r_previous = 0
    for k in tqdm(range(max_iterations)):
        if k == 0:
            X = np.zeros_like(A)
        else:
            sk = r_previous + 1
            (U, S, V) = _my_svd(Y, sk, algorithm)
            while np.min(S) >= tau:
                sk = sk + 5
                (U, S, V) = _my_svd(Y, sk, algorithm)

            shrink_S = np.maximum(S - tau, 0)
            r_previous = np.count_nonzero(shrink_S)
            diag_shrink_S = np.diag(shrink_S)
            X = np.linalg.multi_dot([U, diag_shrink_S, V])
        Y += delta * mask * (A - X)

        if np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A) < epsilon:
            break

    return X

def _my_svd(M, k, algorithm):
    if algorithm == 'randomized':
        (U, S, V) = randomized_svd(M, n_components=min(k, M.shape[1]-1), n_oversamples=20)
    elif algorithm == 'arpack':
        (U, S, V) = svds(M, k=min(k, min(M.shape)-1))
        S = S[::-1]
        U, V = svd_flip(U[:, ::-1], V[::-1])
    else:
        raise ValueError("unknown algorithm")
    return (U, S, V)


if __name__ == "__main__":
    m, n = 200, 160
    k = 10
    noise = 0.1
    mask_prob = 0.75

    U, V, R = gen_factorization_with_noise(m, n, k, noise)
    mask = gen_mask(m, n, mask_prob)

    plt.imshow(R.T)
    plt.title("The matrix to be complete")
    plt.show()

    plt.imshow(mask.T)
    plt.title("The the mask")
    plt.show()

    R_hat = singular_value_thresholding(R, mask)
    print("RMSE:", get_rmse(U, V, R_hat, mask))

    plt.imshow(R_hat.T)
    plt.title("The the matrix after completion")
    plt.show()