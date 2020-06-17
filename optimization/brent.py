# find a minimum of a function via Brent's method
def brent(f, a=-float("inf"), b=+float("inf")):

    _golden = 0.381966011250105097
    x0 = a + _golden * (b - a)
    f0 = f(x0)

    x2 = x1 = x0
    f2 = f1 = f0
    
    d = 0.0
    e = 0.0

    iters = 1
    while True:
        iters += 1

        m = 0.5 * (a + b)
        tol = 10**(-8) * abs(x0) + 10**(-8)
        tol2 = 2.0 * tol

        # Check the stopping criterion.
        if abs(x0 - m) <= tol2 - 0.5 * (b - a):
            break

        p = q = r = 0.0

        # "To be acceptable, the parabolic step must (i) fall within the
        # bounding interval (a, b), and (ii) imply a movement from the best
        # current value x0 that is less than half the movement of the step
        # before last."
        #   - Numerical Recipes 3rd Edition: The Art of Scientific Computing.

        # Compute the polynomial of the least degree (Lagrange polynomial)
        # that goes through (x0, f0), (x1, f1), (x2, f2).
        if tol < abs(e):
            r = (x0 - x1) * (f0 - f2)
            q = (x0 - x2) * (f0 - f1)
            p = (x0 - x2) * q - (x0 - x1) * r
            
            q = 2.0 * (q - r)
            if q > 0.0: p = -p
            q = abs(q)

            r = e
            e = d

        # Take the polynomial interpolation step.
        if abs(p) < abs(0.5 * q * r) and q * (a - x0) < p and p < q * (b - x0):
            d = p / q
            u = x0 + d
            # Function must not be evaluated too close to a or b.
            if (u - a) < tol2 or (b - u) < tol2:
                d = tol if x0 < m else -tol

        # Take the golden-section step.
        else:
            e = b - x0 if x0 < m else a - x0
            d = _golden * e

        # Function must not be evaluated too close to x0.
        if tol <= abs(d):
            u = x0 + d
        elif 0.0 < d:
            u = x0 + tol
        else:
            u = x0 - tol
        fu = f(u)

        if fu <= f0:
            if u < x0: b = x0
            else:      a = x0

            x2, x1 = x1, x0
            f2, f1 = f1, f0
            x0, f0 = u, fu

        else:
            if u < x0: a = u
            else:      b = u
            
            # Insert u between (rank-wise) x0 and x1 in the triple (x0, x1, x2).
            if fu <= f1 or x1 == x0:
                x2, x1 = x1, u
                f2, f1 = f1, fu
            # Insert u in the last position of the triple (x0, x1, x2).
            elif fu <= f2 or x2 == x0 or x2 == x1:
                x2, f2 = u, fu

    return x0, iters

if __name__ == '__main__':
    f = lambda x: x**3 + 3 * x**2 + x - 4
    x, iters = brent(f, -10, 10)
    print("Brent method soln: x = ", x)
    print("Converge in %d iterations " %iters)
    print(f(x))
    