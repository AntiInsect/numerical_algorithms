import numpy as np


class RMSprop:
    
    def __init__(self, fn, fn_grad):
        self.fn = fn
        self.fn_grad = fn_grad
        
    # http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
    def run(self, x_init, y_init, n_iter, lr = 0.001, beta = .9, tol= 1e-5, epsilon = 1e-8):
        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        
        dx_sq = dx**2
        dy_sq = dy**2

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
                
            dx, dy = self.fn_grad(self.fn, x, y)
            
            dx_sq = beta * dx_sq + (1 - beta) * dx * dx
            dy_sq = beta * dy_sq + (1 - beta) * dy * dy

            x += - lr * dx / np.sqrt(dx_sq + epsilon)
            y += - lr * dy / np.sqrt(dy_sq + epsilon)
            
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  RMSprop  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  RMSprop  \033[0m \nDid not converge')
        else:
            print('\033[1m  RMSprop  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_RMSprop = z_path
        self.z_RMSprop = z
        
        return x_path,y_path,z_path
    
    
