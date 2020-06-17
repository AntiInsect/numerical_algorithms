import numpy as np


def successive_approximation(fn, a, b):

#----------------------------------------------------------------------------
# Determines a root x of function Func isolated in [a,b] by the method of
# successive approximations. x contains on entry an initial approximation.
# Error code: 0 - normal execution
#             1 - interval does not contain a root
#             2 - max. number of iterations exceeded
#             3 - diverging process

#----------------------------------------------------------------------------

    x = 0
    dx = -fn(x)    
    for it in range(1, 1000):
        
        f = fn(x)
        if abs(f) > abs(dx):
            break
         
        dx = -f
        x += dx
        if x < a or x > b:
            return x, 1
        if abs(dx) <= 10**(-10):
            return x, 0
        
    if it > 1000:
        print("Iter: max. no. of iterations exceeded !")
        return x, 2
    
    if abs(f) > abs(dx):
        print("Iter: diverging process !")
        return x, 3


if __name__ == "__main__":

    fn = lambda x: x - np.exp(-x)
    a, b = -1e10, 1e10 

    x, flag = successive_approximation(fn, a, b)
    print("final value:", x)
    print("The flag value", flag)