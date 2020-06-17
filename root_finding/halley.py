import random 


def halley(f):
    x = random.random()

    iters = 1
    while abs(f(x)) > 10**(-10):
        iters += 1
        a = 2 * df(x) ** 2
        b = f(x) * ddf(x)
        x -= 2 * f(x) * df(x) / (a - b)

    return x, iters

if __name__ == "__main__":
    fn = lambda x: x**3 - x - 2

    def df(x):
        h = 10**(-10)
        return (fn(x+h) - fn(x)) / h

    def ddf(x):
        h = 10**(-10)
        return (df(x+h) - df(x)) / h

    x, iters = halley(fn)
    print("Bisection method soln: x = ", x)
    print("Converge in %d iterations " %iters)
    print(fn(x))