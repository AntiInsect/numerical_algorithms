import numpy as np

# https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
class momentum:
    
    def __init__(self, fn, fn_grad):
        self.fn = fn
        self.fn_grad = fn_grad
    
    def run(self, x_init, y_init, n_iter, lr, beta, tol= 1e-5):
        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        mu_x = 0
        mu_y = 0

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)

            mu_x = beta * mu_x + lr * dx
            mu_y = beta * mu_y + lr * dy

            x += -mu_x
            y += -mu_y
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Momentum  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Momentum  \033[0m \nDid not converge')
        else:
            print('\033[1m  Momentum  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_momentum = z_path
        self.z_momentum = z
        return x_path,y_path,z_path
    