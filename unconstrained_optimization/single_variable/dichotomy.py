import numpy as np
import matplotlib.pyplot as plt


def dichotomy(goal_function, a, b, epsilon, delta):
    x = 0

    while (b - a) > epsilon:
        lamda = (a + b - delta)/2
        mu = (a + b + delta)/2

        if goal_function(lamda) < goal_function(mu):
            b = mu
            x = lamda
        else:
            a = lamda
            x = mu

    return x


if __name__ == "__main__":
    goal_function = lambda x: x**4-4*x**2-4*x+1

    epsilon = 0.00001
    delta = epsilon / 1000
    a = -10
    b = 10
    n = 100

    x_min_dichotomy = dichotomy(goal_function, a, b, epsilon, delta)
    # print(x_min_dichotomy)
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
        plt.plot([x_min], [goal_function(x_min)], marker='x', markersize=4, color="green", label="dichotomy")
        plt.title("Minimum of F(x) by dichotomy method")
        plt.legend()
        plt.show()
    draw(goal_function, a, b, x_min_dichotomy)
    
