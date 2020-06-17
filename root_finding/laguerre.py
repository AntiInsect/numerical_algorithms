import numpy as np


# numerical recipes 3rd edition
def laguerre_with_roots_polishing(a, polish=False):
    m = len(a) - 1
    roots = np.empty(len(a) - 1, dtype=complex)
    
    # copy coefficients for successful deflation
    ad = a.copy()
    for j in range(m-1, -1, -1):
        x = 0.0 # start at zero to favor convergence to
        # smallest remaining root, and return the root.
        ad_v = np.empty(j+2, dtype=complex)
        for jj in range(j+2):
            ad_v[jj] = ad[jj]
        
        ad_v, x, its = laguerre(ad_v, x)
        if abs(x.imag) <= 2.0 * 10**(-10) * abs(x.real):
            x = x.real + 0.0*1j

        roots[j] = x
        b = ad[j+1]
        for jj in range(j, -1, -1):
            c, ad[jj] = ad[jj], b
            b = x * b + c

    return roots

def laguerre(a, x):
    ad_v = a
    mr = 8
    mt = 10
    maxit = mt * mr
    eps = np.finfo(float).eps
    # EPS here: estimated fractional roundoff error

    # try to break (rare) limit cycles with
    # mr different fractional values, once every mt steps,
    # for maxit total allowed iterations
    frac = [0.0,0.5,0.25,0.75,0.13,0.38,0.62,0.88,1.0]
    m = len(a) - 1
    iters = 1
    for iter in range(1, maxit+1):
        # loop over iterations up to allowed maximum
        its = iter
        b = a[m]
        err = abs(b)
        d = f = 0.0
        abx = abs(x)
        for j in range(m-1, 0-1, -1):
            # efficient computation of the polynomial
            # and its first two derivatives. f stores P''/2
            f = x * f + d
            d = x * d + b
            b = x * b + a[j]
            err = abs(b) + abx * err

        # estimate of roundoff error in evaluating 
        # polynomial
        err *= eps
        if abs(b) <= err: return ad_v, x, its  # we are on the root
        # the generic case: use Laguerre's formula
        g = d/b
        g2 = g**2
        h = g2 - 2.0 * f/b
        sq = (float(m-1) * (float(m)*h - g2))**(1/2)
        gp = g + sq
        gm = g -sq
        abp = abs(gp)
        abm = abs(gm)
        if abp < abm: gp = gm
        if max(abp, abm) > 0.0:
            dx = float(m) / gp
        else:
            # equivalent to polar(1+abx, iter)
            dx = (1+abx) * np.exp(iter*1j)
        x1 = x - dx
        if x == x1:
            return ad_v, its

        if iter % mt != 0:
            x = x1
        else:
            x -= np.frac[int(iter/mt)] * dx

    print('not converged')
    return ad_v, x, its

if __name__ == "__main__":
    degree = 10
    coeff = np.random.randint(-10, 10, size=degree+1) * 1.
    roots = laguerre_with_roots_polishing(coeff)
    for i in range(len(roots)):
        print("Root " + str(i+1) + " : " + str(roots[i]))
