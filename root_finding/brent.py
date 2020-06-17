import numpy as np


# numerical recipes 3rd edition
# Van Wijngaarden-Dekker-Brent method
def brent(f, x1, x2):
    a = x1
    b = x2

    fa = f(a)
    fb = f(b)

    d = 0.0
    e = 0.0
    p = 0.0
    q = 0.0
    r = 0.0
    s = 0.0
    c = 0.0

    tol1 = 0.0
    xm = 0.0

    assert fa * fb <= 0
    fc = fb

    iters = 0
    while True:
        iters += 1

        if fa * fb > 0:
            c = a
            fc = fa
            e = d = b - a
        
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        
        tol1 = 2 * 10**(-10) * abs(b) + 0.5 * 10**(-10)
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            return b, iters
        
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * xm * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * ( 2 * xm * q * (q - r) - (b - a) * (r - 1) )
                q = (q - 1) * (r - 1) * (s - 1)
            
            if p > 0:
                q = -q
            p = abs(p)

            min1 = 3 * xm * q - abs(tol1 * q)
            min2 = abs(e * q)

            min_ = min1 if min1 < min2 else min2

            if 2 * p < min_:
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
    
        a = b
        fa = fb

        if abs(d) > tol1:
            b += d
        else:
            b += np.sign(tol1, xm)
            fb = f(b)


if __name__ == "__main__":
    low, high = 1, 2
    fn = lambda x: x**3 - x - 2
    x, iters= brent(fn, low, high)
    print("Brent method soln: x = ", x)
    print("Converge in %d iterations " %iters)
    print(fn(x))


