import numpy as np


# https://github.com/vrdabomb5717/jenkins_traub
def jenkins_traub(coefficients, epsilon, max_iterations):
    # Remove leading coefficients equal to 0
    while len(coefficients) > 0 and coefficients[0] == 0:
        coefficients.pop(0)

    if len(coefficients) <= 1:
        return

    degree = len(coefficients) - 1
    for x in range(degree):
        if degree > 0:
            _, s = jenkins_traub_inner(coefficients, epsilon, max_iterations)
            coefficients, _ = synthetic_division(coefficients, s)
            yield s


def jenkins_traub_inner(coefficients, epsilon, max_iters):
    """
    Find the smallest root of a polynomial using the Jenkins-Traub algorithm.
    Returns the deflated polynomial and the (approximately) smallest root.
    """

    # Find zero root, if any
    if len(coefficients) > 0 and coefficients[-1] == 0: return coefficients, 0
    if len(coefficients) < 2: return None, None
    if len(coefficients) == 2: return None, -coefficients[-1] / coefficients[0]

    # Scale coefficients so leading coefficient is 1
    a = np.array(coefficients) / coefficients[0]

    powers = np.array(range(len(coefficients)-1, -1, -1))

    # H^0 is the derivative of the polynomial
    H_0 = powers * a
    H_0 = H_0[:-1]

    H_lambda = H_0.copy()
    s_lambda = None

    # Stage 1: accentuate small zeros.
    H_lambda, iters1 = stage1(a, H_0, epsilon, 5)
    print("Done with Stage 1 after {} iterations.".format(iters1))

    # Get s using a modified Cauchy polynomial and Newton's iteration.
    # Modified polynomial has coefficients that are the moduli of the
    # original polynomial, but the last root is -1.
    coeff_modified = np.absolute(a)
    coeff_modified[-1] *= -1
    der_modified = powers * coeff_modified
    der_modified = der_modified[:-1]

    mod_function = evaluate(coeff_modified)
    mod_derivative = evaluate(der_modified)

    # Only get beta up to two decimal places.
    beta = newton_raphson(1, mod_function, mod_derivative, 0.01, 500)

    while True:
        try:
            phi = complex(np.random.random() * 2 * np.pi)
            s_lambda = np.exp(phi * (0+1j)) * beta

            H_lambda, iters2 = stage2(a, H_lambda, s_lambda, epsilon, 100)
            print("Done with Stage 2 after {} iterations.".format(iters2))
            H_lambda, s_lambda, iters3 = stage3(a, H_lambda, s_lambda, epsilon, max_iters)
            print("Done with Stage 3 after {} iterations.".format(iters3))

            return H_lambda, s_lambda

        except ConvergenceError:
            max_iters *= 2
            continue

def stage1(a, H_lambda, epsilon, max_iterations):
    """
    Perform Stage 1, the no-shift process of Jenkins-Traub.
    Returns the deflated polynomial, and the number of iterations the stage took.
    """

    H_bar_lambda = H_lambda / H_lambda[0]

    iters = -1
    for _ in range(max_iterations):
        iters += 1
        try: H_bar_lambda, _ = next_step(a, H_bar_lambda, 0, epsilon, False)
        except RootFound: break

    return H_bar_lambda, iters


def stage2(a, H_lambda, s, epsilon, max_iters):
    """
    Perform Stage 2, the fixed shift process of Jenkins-Traub.
    Returns the deflated polynomial, and the number of iterations the stage took.
    """

    t = t_prev = float('inf')
    H_bar_lambda = H_lambda / H_lambda[0]
    
    iters = 0
    while True:
        t_prev, t_prev_prev = t, t_prev
        try:
            H_bar_lambda, t = next_step(a, H_bar_lambda, s, epsilon)
        except RootFound:
            break

        iters += 1

        condition1 = np.absolute(t_prev - t_prev_prev) <= 0.5 * np.absolute(t_prev_prev)
        condition2 = np.absolute(t - t_prev) <= 0.5 * np.absolute(t_prev)
        condition3 = iters > max_iters

        if (condition1 and condition2) or condition3:
            break
    return H_bar_lambda, iters


def stage3(a, H_L, s, epsilon, max_iterations):
    """
    Perform Stage 3, the variable-shift recurrence of Jenkins-Traub.
    Returns the deflated polynomial, the root, and the number of
    iterations the stage took.
    """
    polynomial = evaluate(a)
    H_bar_coefficients = H_L / H_L[0]
    H_bar = evaluate(H_bar_coefficients)
    s_L = s - polynomial(s) / H_bar(s)

    H_bar_lambda = H_bar_coefficients.copy()
    s_lambda = s_L
    s_lambda_prev = complex('inf')

    num_iterations = 0

    while np.absolute(s_lambda - s_lambda_prev) > epsilon and num_iterations < max_iterations:
        p, p_at_s_lambda = synthetic_division(a, s_lambda)
        h_bar, h_bar_at_s_lambda = synthetic_division(H_bar_lambda, s_lambda)
        h_bar = np.insert(h_bar, 0, 0)  # watch for polynomial length differences

        # If we found a root, short circuit the other logic.
        # Used for when we found a root exactly, and we'd end up dividing
        # by 0 when finding s_lambda
        if np.absolute(p_at_s_lambda) < epsilon:
            return H_bar_lambda, s_lambda, num_iterations

        H_bar_lambda_next = p - (p_at_s_lambda / h_bar_at_s_lambda) * h_bar
        H_bar_next = evaluate(H_bar_lambda_next)

        num_iterations += 1

        s_lambda, s_lambda_prev = (s_lambda - p_at_s_lambda / H_bar_next(s_lambda)), s_lambda
        H_bar_lambda = H_bar_lambda_next

    if num_iterations >= max_iterations:
        print('Stage 3 could not converge after {} iterations'.format(num_iterations))
        raise ConvergenceError()

    return H_bar_lambda, s_lambda, num_iterations


class ConvergenceError(Exception):
    pass

class RootFound(Exception):
    pass

def evaluate(coeff):
    """
    Evaluate a polynomial with given coefficients in highest to lowest order.
    """
    return lambda s: synthetic_division(coeff, s)[1]


def newton_raphson(x_0, function, first_derivative, epsilon, max_iterations):
    """
    Find the root of a polynomial using Newton-Raphson iteration.
    """
    x_i = x_0
    num_iterations = 0
    x_i_next = x_i - (function(x_i) / first_derivative(x_i))
    num_iterations += 1

    while abs(x_i_next - x_i) > epsilon and num_iterations < max_iterations:
        x_i = x_i_next
        x_i_next = x_i - (function(x_i) / first_derivative(x_i))
        num_iterations += 1
    return x_i


def synthetic_division(coefficients, s):
    """
    Perform synthetic division and evaluate a polynomial at s.
    """
    deflated = np.empty(len(coefficients) - 1, dtype=np.complex128)
    deflated[0] = coefficients[0]
    for i, a_i in enumerate(coefficients[1:-1], start=1):
        deflated[i] = a_i + deflated[i - 1] * s

    evaluation = coefficients[-1] + deflated[-1] * s
    return deflated, evaluation


def next_step(a, H_bar_lambda, s, epsilon, generate_t=True):
    """
    Generate the next H_bar_lambda and t, if desired.
    """
    p, p_at_s = synthetic_division(a, s)
    h_bar, h_bar_at_s = synthetic_division(H_bar_lambda, s)

    # If we found a root, short circuit the other logic.
    # Used for when we found a root exactly, and we'd end up dividing
    # by 0 when finding s_lambda
    if np.absolute(p_at_s) < epsilon:
        raise RootFound()

    if np.absolute(h_bar_at_s) < epsilon:
        h_bar_at_s += epsilon / 100

    t = None
    if generate_t:
        t = s - p_at_s / h_bar_at_s

    # watch for polynomial length differences
    h_bar = np.insert(h_bar, 0, 0)  
    return p - (p_at_s / h_bar_at_s) * h_bar, t

if __name__ == '__main__':
    coefficients = [1, -9.01, 27.08, -41.19, 32.22, -10.1]
    epsilon = 1.0e-10
    max_iterations = 10000

    roots = [x for x in jenkins_traub(coefficients, epsilon, max_iterations)]
    for i in range(len(roots)):
        print("Root " + str(i+1) + " : " + str(roots[i]))
