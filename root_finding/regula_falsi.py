import math


# Or the false position method
def regula_falsi(f, x0, x1):
    assert f(x0) * f(x1) < 0

    x = 0 
    iters = 1
    while abs(f(x)) > 10**(-5):
        iters += 1
        x = x1 - (f(x1)*(x1-x0)) / (f(x1)-f(x0))

        if f(x0) * f(x) < 0:
            x1 = x
        else:
            x0 = x

    return x, iters


if __name__ == "__main__":
    x0 = 1
    x1 = 2
    fn = lambda x: x**3 - x - 2
    x, iters = regula_falsi(fn, x0, x1)
    print("Regula falsi method soln: x = ", x)
    print("Converge in %d iterations " %iters)
    print(x**3 - x - 2) 
