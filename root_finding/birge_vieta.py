import numpy as np


# only get the real roots
def birge_vieta(coeff):
    order = len(coeff) - 1
    if order <= 1: return None
    roots = np.zeros(order)

    nx = 0
    x = 0
    for m in range(order, 1, -1):
        for it in range(1, 1000):
            p = coeff[0]
            d = p
            for j in range(1, m):
                p = p * x + coeff[j]
                d = d * x + p
            p = p*x + coeff[m]
            d = -p/d if d else -p
            x += d

            if abs(d) <= 10**(-10):
                break
        
        nx = 0
        roots[nx] = x

        for j in range(1,m):
            coeff[j] += coeff[j-1] * x
    
    nx += 1
    roots[nx] = - coeff[1] / coeff[0]

    return roots[:nx+1]


if __name__ == "__main__":
    order = 10
    coeff = np.random.randint(-10, 10, size=order+1) * 1.
    print(coeff)
    roots = birge_vieta(coeff)
    for i in range(len(roots)):
        print("Root " + str(i+1) + " : " + str(roots[i]))


