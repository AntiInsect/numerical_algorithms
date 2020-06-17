import numpy as np


def lanczos(A, v0, k):
    print('LANCZOS METHOD !')
    V = np.mat( v0.copy() / np.linalg.norm(v0) )
    alpha =  np.zeros(k)  
    beta =  np.zeros(k+1)  
    for m in range(k):
        vt =  A @  V[ :, m]
        
        if m > 0:
            vt -= beta[m] * V[:, m-1]
        
        alpha[m] = (V[:, m].H * vt )[0, 0]
        vt -= alpha[m] * V[:, m]
        beta[m+1] = np.linalg.norm(vt)
        V =  np.hstack( (V, vt.copy() / beta[m+1] ) ) 
    rbeta = beta[1:-1]    
    H = np.diag(alpha) + np.diag(rbeta, 1) + np.diag(rbeta, -1)
    return V,  H

def orthogonal_lanczos(A, v0, k):
    print('FULL ORTHOGONAL LANCZOS METHOD')
    from numpy import finfo, sqrt
    reps = 10*sqrt(finfo(float).eps)
    V = np.mat( v0.copy() / np.linalg.norm(v0) )
    alpha =  np.zeros(k)  
    beta =  np.zeros(k+1)  
    
    for m in range(k):
        vt = A @ V[ :, m]
        if m > 0:
            vt -= beta[m] * V[:, m-1]

        alpha[m] = (V[:, m].H * vt )[0, 0]
        vt -= alpha[m] * V[:, m]
        beta[m+1] = np.linalg.norm(vt)
        
        # reortogonalization
        h1 = V.H @ vt
        vt -= V @ h1

        if np.linalg.norm(h1) > reps:
            vt -= V @ V.H @ vt

        V =  np.hstack( (V, vt.copy() / beta[m+1] ) ) 
    rbeta = beta[1:-1]  

    H = np.diag(alpha) + np.diag(rbeta, 1) + np.diag(rbeta, -1)
    return V,  H

if __name__ == "__main__":

    N = 5
    A = [[7, 2.5, 2, 1.5, 1],
         [2.5, 8, 2.5, 2, 1.5],
         [2, 2.5, 9, 2.5, 2],
         [1.5, 2, 2.5, 10, 2.5],
         [1, 1.5, 2, 2.5, 11]]
    A = np.matrix(A)
    v = np.mat( np.ones(N) ).T
    V, h = lanczos(A, v, N)
    V, h = orthogonal_lanczos(A, v, N)
