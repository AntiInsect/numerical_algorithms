import math

def secant(f, x1, x2):
    error = 1
    iters = 1
    while error > 10**(-5):
        iters += 1

        x = x2 - (f(x2)*(x2-x1)) / (f(x2)-f(x1))
        error = abs(x - x2)
        x1, x2 = x2, x

    return x, iters


if __name__ == '__main__':
    f = lambda x: x**3 - x - 2
    x, iters = secant(f, 0, math.pi/2)
    print("Regula falsi method soln: x = ", x)
    print("Converge in %d iterations " %iters)
    print(x**3 - x - 2)