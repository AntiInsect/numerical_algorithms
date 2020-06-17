import cmath
import random
import numpy as np


def bairstow(coeff, r0_guess=None, r1_guess=None, roots=None):
    if roots    is None: roots = []
    if r0_guess is None: r0_guess = random.random()
    if r1_guess is None: r1_guess = random.random()

    order = len(coeff) - 1
    if order < 1:
        return roots
    elif order == 1:
        roots.append(- coeff[0] / coeff[1])
        return roots
    elif order == 2:
        D = coeff[1]**2 - 4 * coeff[2] * coeff[0]
        roots.append((-coeff[1] - cmath.sqrt(D)) / (2.0 * coeff[2]))
        roots.append((-coeff[1] + cmath.sqrt(D)) / (2.0 * coeff[2]))
        return roots

    def apply_guess(coeff):
        n = len(coeff)
        guess = np.zeros_like(coeff)
        guess[n-1] = coeff[n-1]
        guess[n-2] = coeff[n-2] + r0_guess * guess[n-1]
        for i in range(n-3, -1, -1):
            guess[i] = coeff[i] + r0_guess * guess[i+1] + r1_guess * guess[i+2]
        return guess

    b = apply_guess(coeff)
    c = apply_guess(b)

    D = ((c[2]*c[2]) - (c[3]*c[1])) ** (-1.0)
    r0_guess += D * (  (c[2]) * (-b[1]) + (-c[3]) * (-b[0]))
    r1_guess += D * ( (-c[1]) * (-b[1]) +  (c[2]) * (-b[0]))
    
    if abs(b[0]) > 10**(-14) or abs(b[1]) > 10**(-14):
        return bairstow(coeff, r0_guess, r1_guess, roots)
        
    if order >= 3:
        D = ((-r0_guess)**(2.0))-((4.0)*(1.0)*(-r1_guess))
        roots.append((r0_guess - (cmath.sqrt(D))) / 2)
        roots.append((r0_guess + (cmath.sqrt(D))) / 2)
        return bairstow(b[2:], r0_guess, r1_guess, roots)


if __name__ == "__main__":
    order = 10
    coeff = np.random.randint(-10, 10, size=order+1) * 1.
    roots = bairstow(coeff)
    for i in range(len(roots)):
        print("Root " + str(i+1) + " : " + str(roots[i]))
