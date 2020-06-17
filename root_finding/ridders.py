import math
import numpy as np


def ridder(f, a, b):
    fa = f(a)
    fb = f(b)

    if fa == 0.0: return a
    if fb == 0.0: return b
    
    if np.sign(fa) == np.sign(fb):
        print('Root is not bracketed')


    xold = 0
    iters = 1
    while True:
        iters += 1
        
        c = (a + b) / 2
        fc = f(c)

        s = np.sqrt(fc**2 - fa*fb)
        
        dx = (c - a) * fc / (s + 10**(-5))

        if (fa - fb) < 0.0:
            dx = -dx 
        
        x = c + dx
        fx = f(x)

        if abs(x - xold) < 10**(-9) * max(abs(x), 1.0):
            return x, iters
        xold = x
        
        # Re-bracket the root as tightly as possible
        if np.sign(fc) == np.sign(fx): 
            if np.sign(fa) != np.sign(fx):
                b = x
                fb = fx
            else:
                a = x
                fa = fx
        else:
            a, fa = c, fc
            b, fb = x, fx
    
    print('Does not converge ')

if __name__ == "__main__":
    low, high = 1, 2
    fn = lambda x: x**3 - x - 2
    x, iters = ridder(fn, low, high)
    print("Bisection method soln: x = ", x)
    print("Converge in %d iterations " %iters)
    print(fn(x))

