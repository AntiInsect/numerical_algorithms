import cmath


def muller(f, x0, x1, x2):
    iters = 0
    while abs(f(x2)) > 10**(-7):
        q = (x2 - x1) / (x1 - x0)
        a = q*f(x2) - q*(1+q)*f(x1) + q**2*f(x0)
        b = (2*q + 1)*f(x2) - (1+q)**2*f(x1) + q**2*f(x0)
        c = (1 + q)*f(x2)
        r = x2 - (x2 - x1)*((2*c)/(b + cmath.sqrt(b**2 - 4*a*c)))
        s = x2 - (x2 - x1)*((2*c)/(b - cmath.sqrt(b**2 - 4*a*c)))
    
        x = r if abs(f(r)) < abs(f(s)) else s

        if x.imag == 0j:
            x = x.real
        
        x0 = x1
        x1 = x2
        x2 = x
        iters += 1

    # when root is complex double check complex conjugate
    if isinstance(x, complex):
	    conjugate = complex(x.real, -x.imag)
	    if abs(f(conjugate)) < 10**(-7):
		    print("and \t{:.4f}".format(conjugate)+"\t{:.4f}".format(f(conjugate)))

    return x, iters


if __name__ == "__main__":

    fn = lambda x: x**3 - x**2 - x - 1	

    x0, x1, x2 = 0, 1, 2
    x, iters = muller(fn, x0, x1, x2)
    print("final value:", x)
    print("Converge in %d iterations " %iters)

