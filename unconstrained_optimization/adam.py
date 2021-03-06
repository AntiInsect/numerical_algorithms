import numpy as np


# adaptive learning rate
class adam:    

    def __init__(self, fn, fn_grad):
        self.fn = fn
        self.fn_grad = fn_grad

    # http://ruder.io/optimizing-gradient-descent/index.html#adam
    # https://arxiv.org/abs/1412.6980
    def run(self, x_init, y_init, n_iter, lr, beta_1 = .9, beta_2 = .99, tol= 1e-5, epsilon = 1e-8):
        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        m_x = 0
        v_x = 0
        m_y = 0
        v_y = 0

        lr_t = lr * (np.sqrt(1 - beta_2))/(1 - beta_1)

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)

            m_x = beta_1 * m_x + dx * (1 - beta_1)
            v_x = beta_2 * v_x + dx * dx * (1 - beta_2)
            m_x_hat = m_x / (1 - np.power(beta_1, max(i, 1)))
            v_x_hat = v_x / (1 - np.power(beta_2, max(i, 1)))

            m_y = beta_1 * m_y + dy * (1 - beta_1)
            v_y = beta_2 * v_y + dy * dy * (1 - beta_2)
            m_y_hat = m_y / (1 - np.power(beta_1, max(i, 1)))
            v_y_hat = v_y / (1 - np.power(beta_2, max(i, 1)))

            x += - (lr_t * m_x_hat)/(np.sqrt(v_x_hat) + epsilon)
            y += - (lr_t * m_y_hat)/(np.sqrt(v_y_hat) + epsilon)
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Adam  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Adam  \033[0m \nDid not converge')
        else:
            print('\033[1m  Adam  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_adam = z_path
        self.z_adam = z
        return x_path,y_path,z_path
