import numpy as np
import matplotlib.image as img
from tqdm import tqdm 


def evalpoly(p, root):
    return np.dot(p, np.power(root, range(p.shape[0])))

def durand_kerner(p):
    tol = 10**(-10)
    degree = p.shape[0]-1

    # initial guess of roots
    roots = np.power(0.5+0.5j, np.arange(degree))

    iters = 1
    while True:
        iters += 1
        W = np.zeros(p.shape[0]-1,dtype=complex)
        for k in range(degree):
           aux = 1.0
           for l in range(degree):
               if k != l:
                   aux *= roots[k] - roots[l]
           W[k] = evalpoly(p, roots[k])/aux
        roots -= W

        if iters > 5000:
            print("Bad random, run again")
            exit(0)

        if np.linalg.norm(W) < tol:
            break

    for i in range(degree):
        c = 1
        root = roots[i]
        for j in range(i+1, degree):
            if np.abs(roots[j] - roots[i]) < tol:
                c += 1
                root += roots[j]
        if c > 1:
            root /= c
            for j in range(i+1,degree):
                if np.abs(roots[j] - roots[i]) < tol:
                    roots[j] = root
            roots[i] = root
    return roots, iters

if __name__ == "__main__":
    # [ 1.  1.  4.  2. -4.  0.  1.]
    # [ 3. -4.  2.  2.  2.  0.  3. -4. -1. -2.  1.]
    # [ 0. -1.  1.]
    roots, iters = durand_kerner(np.array([3., -4. ,2. ,2. ,2., 0. , 3., -4., -1., -2., 1.]))
    print("Converge use %d iterations" %iters)
    for i in range(len(roots)):
        print("Root " + str(i+1) + " : " + str(roots[i]))
