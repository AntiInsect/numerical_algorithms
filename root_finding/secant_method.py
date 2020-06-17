import math

def secant_method(f, x1, x2):
    error = 1

    iterations = 1
    while error > 10**(-5):
        iterations += 1

        x_new = x2 - (f(x2)*(x2-x1))/(f(x2)-f(x1))
        error = abs(x_new - x2)
        x1 = x2
        x2 = x_new
        print(f'x{iterations+1}: {x2}')

    print(f'Result: {x2}\nIterations: {iterations}')

if __name__ == '__main__':
    f = lambda x: x - math.cos(x)
    secant_method(f, 0, math.pi/2)