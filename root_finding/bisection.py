import math


def bisection(f, low, high):
    assert f(low) * f(high) < 0
    
    iters = 1
    while True:
        iters += 1
        mid = (high + low) / 2.0

        if f(mid) == 0 or (high - low) / 2.0 < 10**(-5):
            return mid, iters

        if f(mid) * f(low) > 0:
            low = mid
        else:
            high = mid
            
    return None, -1


if __name__ == "__main__":
    low, high = 1, 2
    fn = lambda x: x**3 - x - 2
    x, iters = bisection(fn, low, high)
    print("Bisection method soln: x = ", x)
    print("Converge in %d iterations " %iters)
    print(fn(x))
