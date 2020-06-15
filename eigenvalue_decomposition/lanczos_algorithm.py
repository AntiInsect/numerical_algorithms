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

    def ConstructMatrix(N):
        H = np.mat(np.zeros([N,N]))
        for i in range(N):
            for j in range(N):
                H[i, j] = float(1 + min(i, j))
        return H

    N = 10
    A = ConstructMatrix(N)

    v = np.mat( np.ones(N) ).T
    V, h = lanczos(A, v, N)
    V, h = orthogonal_lanczos(A, v, N)
