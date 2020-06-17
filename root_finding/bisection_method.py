import math


def bisection_method(f, low, high):
    assert f(low) * f(high) < 0
    
    for i in range(10000):
        mid = (high + low) / 2.0

        if f(mid) == 0 or (high - low) / 2.0 < 10**(-5):
            return mid, i

        if f(mid) * f(low) > 0:
            low = mid
        else:
            high = mid
            
    return None, -1


if __name__ == "__main__":
    low = 1
    high = 2

    fn = lambda x: math.pow(x,3) - x - 2
    x, iters = bisection_method(fn, low, high)
    print("Bisection method soln: x = ", x)
    print("Converge in %d iterations " %iters)
