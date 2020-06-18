import numpy as np
from tqdm import tqdm


# stochastic gradient descent
def stochastic_gradient_descent(R, k, alpha=0.1, beta=0.01, max_iterations=10000):
    num_users, num_items = R.shape

    # Initialize user and item latent feature matrice
    P = np.random.normal(scale=1./k, size=(num_users, k))
    Q = np.random.normal(scale=1./k, size=(num_items, k))
    
    # Initialize the biases
    b_u = np.zeros(num_users)
    b_i = np.zeros(num_items)
    b = np.mean(R[np.where(R != 0)])
        
    # Create a list of training samples
    samples = [(i, j, R[i, j])
               for i in range(num_users) for j in range(num_items)
               if R[i, j] > 0]
        
    # Perform stochastic gradient descent for number of iterations
    for _ in tqdm(range(max_iterations)):

        np.random.shuffle(samples)
        
        for i, j, r in samples:
            # Computer prediction and error
            prediction = b + b_u[i] + b_i[j] + P[i, :] @ Q[j, :].T
            e = (r - prediction)
            
            # Update biases
            b_u[i] += alpha * (e - beta * b_u[i])
            b_i[j] += alpha * (e - beta * b_i[j])
            
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = P[i, :][:]
            
            # Update user and item latent feature matrices
            P[i, :] += alpha * (e * Q[j, :] - beta * P[i,:])
            Q[j, :] += alpha * (e * P_i - beta * Q[j,:])

        # Compute the mse
        xs, ys = R.nonzero()
        predicted = b + b_u[:,np.newaxis] + b_i[np.newaxis:,] + P @ Q.T
        error = 0
        for x, y in zip(xs, ys):
            error += pow(R[x, y] - predicted[x, y], 2)
        if np.sqrt(error) <= 10**(-10):
            break
    
    X = b + b_u[:,np.newaxis] + b_i[np.newaxis:,] + P @ Q.T

    return X


if __name__ == "__main__":
    k = 2
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    X = stochastic_gradient_descent(R, k)
    print(X)