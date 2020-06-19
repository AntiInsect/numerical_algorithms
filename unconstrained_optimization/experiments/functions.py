import autograd.numpy as np
from autograd import elementwise_grad

# Holder Table Function
holder_fn = lambda x,y: -abs(np.sin(x)*np.cos(y)*np.exp(abs(1-np.sqrt(x**2+y**2)/np.pi)))

# Ackley Function
a,b,c = 20,0.2,2*np.pi
ackley_fn = lambda x,y: -a*np.exp(-b*np.sqrt(1/2*(x**2+y**2)))-np.exp(1/2*(np.cos(c*x)+np.cos(c*y)))+a+np.exp(1)

# Noisy Function
noisy_fn = lambda x,y : np.sin(x**2) * np.cos(3 * y**2) * np.exp(-(x * y) * (x * y)) - np.exp(-(x + y) * (x + y))

# Six-Hump Camel Function
sixhump_fn = lambda x,y : (4 - 2.1*x**2 + ((x**4)/3)) * x**2 + x * y + (-4 + 4 * y**2) * y**2

# Shubert Function
def shubert_fn(x_1, x_2):
    sum1 = 0
    sum2 = 0
    
    for i in range(1,6):
        new1 = i * np.cos(i + 1) * x_1 + i
        new2 = i * np.cos(i + 1) * x_2 + i
        
        sum1 = sum1 + new1
        sum2 = sum2 + new2
        
    y = sum1 * sum2
    
    return y


def fn_grad(fn, x1, x2):
    dy_dx1 = elementwise_grad(fn, argnum=0)(x1, x2)
    dy_dx2 = elementwise_grad(fn, argnum=1)(x1, x2)
    
    return dy_dx1, dy_dx2



