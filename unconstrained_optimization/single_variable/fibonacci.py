import numpy as np
import matplotlib.pyplot as plt


PHI = 1.6180339887499
PSI = -0.61803398871499
REVERSE_SQUARE_ROOT_FIVE = 0.447213595499958


def fibonacci_number(n):
    return (PHI**n - PSI**n)*REVERSE_SQUARE_ROOT_FIVE


def fibonacci(goal_function, a, b, epsilon):
    x = 0
    n = 100

    while (b - a) > epsilon or n != 1:
        lamda = a + (b - a)*(fibonacci_number(n-2)/fibonacci_number(n))
        mu = a + (b - a)*(fibonacci_number(n-1)/fibonacci_number(n))

        if goal_function(lamda) <= goal_function(mu):
            b = mu
            x = lamda
        else:
            a = lamda
            x = mu

        n -= 1

    if n == 1:
        return (mu + lamda) / 2

    return x

if __name__ == "__main__":
    goal_function = lambda x: x**4-4*x**2-4*x+1

    epsilon = 0.00001
    delta = epsilon / 1000
    a = -10
    b = 10
    n = 100

    x_min_fibonacci = fibonacci(goal_function, a, b, epsilon)
    
    # print(x_min_fibonacci)
    def draw(goal_function, a, b, x_min):
        x = np.linspace(a, b, 100)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.plot(x, goal_function(x), color="violet")
        plt.plot([x_min], [goal_function(x_min)], marker='s', markersize=8, color="red", label="fibonacci")
        plt.title("Minimum of F(x) by dichotomy method")
        plt.legend()
        plt.show()
    draw(goal_function, a, b, x_min_fibonacci)