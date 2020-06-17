import math
import random
import numpy as np


def aberth_ehrlich(coeff):
    roots = init_roots(coeff)
    
    iters = 1
    while True:

        valid = 0
        for k, r in enumerate(roots):
            ratio =  image(coeff, r) / derivative(coeff, r)
            offset = ratio / (1 - (ratio * sum(1/(r - x) for j, x in enumerate(roots) if j != k)))
            
            if round(offset.real, 14) == 0 and round(offset.imag, 14) == 0:
                valid += 1
            roots[k] -= offset
        
        if valid == len(roots):
            break

        iters += 1

    return [complex(round(r.real, 12), round(r.imag, 12)) for r in roots], iters

def get_bounds(coeff):
    degree = len(coeff) - 1
    upper = 1 + 1 / abs(coeff[-1]) * max(abs(coeff[x]) for x in range(degree))
    lower = abs(coeff[0]) / (abs(coeff[0]) + max(abs(coeff[x]) for x in range(1, degree + 1)))
    return upper, lower

def init_roots(coeff):
    degree = len(coeff) - 1
    upper, lower = get_bounds(coeff)

    roots = []
    for i in range(degree):
        radius = random.uniform(lower, upper)
        angle = random.uniform(0, math.pi*2)
        root = complex(radius * math.cos(angle), radius * math.sin(angle))
        roots.append(root)

    return roots

# the image of the polynomial
def image(coeff, x):
    return sum(coef*(pow(x, i)) for i, coef in enumerate(coeff))

def derivative(coeff, x):
    return (image(coeff, x + 1.e-12) - image(coeff, x)) / 1.e-12


if __name__ == "__main__":
    degree = 10
    coeff = np.random.randint(-10, 10, size=degree+1) * 1.
    roots, _ = aberth_ehrlich(coeff)
    for i in range(len(roots)):
        print("Root " + str(i+1) + " : " + str(roots[i]))
    