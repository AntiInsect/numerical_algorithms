import numpy as np
import scipy.linalg as sla


def broyden(x, y, fs, Js): 
    f = fs(x, y)
    J = Js(x, y)
 
    iters = 1
    while np.linalg.norm(f) > 10**(-10):
        iters += 1
 
        s = sla.solve(J, -1*f)
 
        x += s[0]
        y += s[1]
        
        newf = fs(x,y)
        J += (np.outer ((newf - f - np.dot(J,s)),s)) / (np.dot(s,s))
        f = newf
 
    return x, y, iters
 
if __name__ == "__main__":
 
    # funtion 
    def f(x,y):
        return np.array([x + 2.*y - 2., x**2. + 4.*y**2. - 4.])
    
    # Jaboci
    def J(x,y):
        return np.array([[1., 2.], [2., 16.]])

    # init guess
    x0, y0 = 1., 2.
    x, y, n = broyden(x0, y0, f, J)
    print("x and y: ", x, y)
    print("iterations: ", n)