import numpy as np 


def arnoldi(A, v0, k):
    print('ARNOLDI METHOD')
    inputtype = A.dtype.type
    V = np.mat( v0.copy() / np.linalg.norm(v0), dtype=inputtype)
    H = np.mat( np.zeros((k+1,k), dtype=inputtype) )
    for m in range(k):
        vt = A * V[:, m]
        for j in range(m+1):
            H[j, m] = (V[:, j].H * vt )[0,0]
            vt -= H[j, m] * V[:, j]
        H[ m+1, m] = np.linalg.norm(vt)
        if m is not k-1:
            V =  np.hstack( (V, vt.copy() / H[ m+1, m] ) ) 
    return V,  H

def arnoldi_fast(A, v0, k):
    print('ARNOLDI FAST METHOD')
    inputtype = A.dtype.type
    V = np.mat(np.zeros((v0.shape[0],k+1), dtype=inputtype))
    V[:,0] = v0.copy()/np.linalg.norm(v0)
    H = np.mat( np.zeros((k+1,k), dtype=inputtype) )
    for m in range(k):
        vt = A*V[ :, m]
        for j in range( m+1):
            H[ j, m] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, m] * V[:, j]
        H[ m+1, m] = np.linalg.norm(vt)
        V[:,m+1] = vt.copy() / H[ m+1, m]
    return V,  H

def arnoldi_fast_nocopy(A, v0, k):
    #  Uses in-place computations and row-major format
    print('ARNOLDI FAST NOCOPY METHOD')
    inputtype = A.dtype.type
    n = v0.shape[0]
    V = np.zeros((k+1,n), dtype=inputtype)
    V[0,:] = v0.T.copy()/np.linalg.norm(v0)
    H = np.zeros((k+1,k), dtype=inputtype)
    for m in range(k):
        V[m+1,:] = np.dot(A,V[m,:])
        for j in range( m+1):
            H[ j, m] = np.dot(V[j,:], V[m+1,:])
            V[m+1,:] -= H[ j, m] * V[j,:]
        H[ m+1, m] = np.linalg.norm(V[m+1,:])
        V[m+1,:] /= H[ m+1, m]
    return V.T,  H

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
    V, h = arnoldi(A, v, N)
    V, h = arnoldi_fast(A, v, N)
    V, h = arnoldi_fast_nocopy(A, v, N)
