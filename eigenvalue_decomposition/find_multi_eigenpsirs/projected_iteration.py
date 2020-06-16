import numpy as np

# Sequentially find the k eigenpairs of a symmetric matrix
def projected_iteration(A, k):
    eigvecs = np.zeros((A.shape[0], k))
    eigvals = np.zeros(A.shape[0])
    for i in range(k):
        v = np.random.randn(A.shape[0])
        for _ in range(1000):
            proj_sum = np.zeros_like(v).reshape(v.shape[0])
            for j in range(i):
                proj_sum += projection(v, eigvecs[:, j])
            
            v -= proj_sum
            v_new = A @ v
            v_new = v_new / np.linalg.norm(v_new)

            if np.all(np.abs(v_new - v) < 1e-8):
                break
            v = v_new
            # print('sss')
            # print(v.shape)

        e = (v.T @ A @ v) / (v.T @ v)
        
        # store eigenpair
        eigvecs[:, i] = np.array(v)
        eigvals[i] = e

    return eigvals, eigvecs

# the projection of b on a
def projection(b, a):
    b = b.reshape(b.shape[0])
    a = a.reshape(a.shape[0])
    return ((a @ b) / (a.T @ a)) * a

if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
        [2.5, 8, 2.5, 2, 1.5],
        [2, 2.5, 9, 2.5, 2],
        [1.5, 2, 2.5, 10, 2.5],
        [1, 1.5, 2, 2.5, 11]]
    matrix = np.array(A)
    eigvals, eigvecs = projected_iteration(matrix, 3)
    print(eigvals)
    print(eigvecs)
