import numpy as np
import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm 


def Jacobi(A):
    n     = A.shape[0]            # matrix size #columns = #lines
    maxit = 100                   # maximum number of iterations
    eps   = 1.0e-15               # accuracy goal
    pi    = np.pi        
    info  = 0                     # return flag
    ev    = np.zeros(n,float)     # initialize eigenvalues
    U     = np.zeros((n,n),float) # initialize eigenvector
    for i in range(0,n): U[i,i] = 1.0

    for t in range(0,maxit):
         s = 0;    # compute sum of off-diagonal elements in A(i,j)
         for i in range(0,n): s = s + np.sum(np.abs(A[i,(i+1):n]))
         if (s < eps): # diagonal form reached
              info = t
              for i in range(0,n):ev[i] = A[i,i]
              break
         else:
              limit = s/(n*(n-1)/2.0)       # average value of off-diagonal elements
              for i in range(0,n-1):       # loop over lines of matrix
                   for j in range(i+1,n):  # loop over columns of matrix
                       if (np.abs(A[i,j]) > limit):      # determine (ij) such that |A(i,j)| larger than average 
                                                         # value of off-diagonal elements
                           denom = A[i,i] - A[j,j]       
                           if (np.abs(denom) < eps): phi = pi/4         
                           else: phi = 0.5*np.arctan(2.0*A[i,j]/denom)  
                           si = np.sin(phi)
                           co = np.cos(phi)
                           for k in range(i+1,j):
                               store  = A[i,k]
                               A[i,k] = A[i,k]*co + A[k,j]*si 
                               A[k,j] = A[k,j]*co - store *si 
                           for k in range(j+1,n):
                               store  = A[i,k]
                               A[i,k] = A[i,k]*co + A[j,k]*si 
                               A[j,k] = A[j,k]*co - store *si 
                           for k in range(0,i):
                               store  = A[k,i]
                               A[k,i] = A[k,i]*co + A[k,j]*si
                               A[k,j] = A[k,j]*co - store *si
                           store = A[i,i]
                           A[i,i] = A[i,i]*co*co + 2.0*A[i,j]*co*si +A[j,j]*si*si 
                           A[j,j] = A[j,j]*co*co - 2.0*A[i,j]*co*si +store *si*si
                           A[i,j] = 0.0                                           
                           for k in range(n):
                                store  = U[k,j]
                                U[k,j] = U[k,j]*co - U[k,i]*si 
                                U[k,i] = U[k,i]*co + store *si 
         info = -t # in case no convergence is reached set info to a negative value "-t"
    return ev,U,t

def compare():
    # MAIN PROGRAM - part (b)
    Nlist = [10, 50, 100]     # range for matrix sizes
    t1    = np.zeros(len(Nlist),float)  # array for timing results of eig-routine
    t2    = np.zeros(len(Nlist),float)  # array for timing results of Jacobi-routine
    count = 0
    for n in tqdm(Nlist):
        A = np.random.rand(n,n)         # matrix of random numbers
        A = 0.5*(A + np.transpose(A))   # symmetrize matrix

        start_time = timeit.default_timer()
        ev1,U      = np.linalg.eig(A)   # eig routine 
        t1[count]  = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        ev2,U,t    = Jacobi(A)
        t2[count]  = timeit.default_timer() - start_time

        count = count + 1

    # plot current results
    plt.loglog(Nlist,t1,'rx-' ,label='numpy.linalg.eig')
    plt.loglog(Nlist,t2,'b+--',label='Jacobi [Python]')

    # format plots
    plt.title('Performace of eigenvalue solvers')
    plt.xlabel('Matrix Size N')
    plt.ylabel('CPU time (sec)')
    plt.legend(loc='lower right', shadow=False, fontsize=12)
    plt.show()  

if __name__ == "__main__":
    A = [[7, 2.5, 2, 1.5, 1],
        [2.5, 8, 2.5, 2, 1.5],
        [2, 2.5, 9, 2.5, 2],
        [1.5, 2, 2.5, 10, 2.5],
        [1, 1.5, 2, 2.5, 11]]
    
    A = np.matrix(A)
    
    ev, U, t = Jacobi(A)
    print("JACOBI METHOD: Number of rotations = ",t)
    print("Eigenvalues  = ",ev)
    print("Eigenvectors = ", U)
