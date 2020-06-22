import numpy as np


class gradient_descent:
    
    def __init__(self, fn, fn_grad):
        self.fn = fn
        self.fn_grad = fn_grad
        
    def run(self, x_init, y_init, n_iter, lr, tol= 1e-5):
        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)
            x += -lr * dx
            y += -lr * dy
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Plain Vanilla  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Plain Vanilla  \033[0m \nDid not converge')
        else:
            print('\033[1m  Plain Vanilla  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_pv = z_path
        self.z_pv = z
        return x_path,y_path,z_path

    
    

    
    
