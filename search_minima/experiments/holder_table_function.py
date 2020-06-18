import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


from IPython.display import HTML

from functions import *
from animations import *

from search_minima.plain_gradient_descent import plain_gradient_descent
from search_minima.momentum import momentum
from search_minima.adam import adam
from search_minima.adamax import adamax
from search_minima.nadam import nadam
from search_minima.amsgrad import amsgrad
from search_minima.nag import nag
from search_minima.RMSprop import RMSprop



# Set Parameters and return array of x,y,z
xmin, xmax, xstep = -1, 3, 0.1
ymin, ymax, ystep = -1.5, 2, 0.1
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = holder_fn(x, y)

x_init = 0.6
y_init = 1.5
n_iter = 2000
lr = 0.015

# Initiate gradient descent 
plain_gradient_descent_ = plain_gradient_descent(fn = holder_fn, fn_grad=fn_grad)
momentum_ = momentum(fn = holder_fn, fn_grad=fn_grad)
adam_ = adam(fn = holder_fn, fn_grad=fn_grad)
adamax_ = adamax(fn = holder_fn, fn_grad=fn_grad)
nadam_ = nadam(fn = holder_fn, fn_grad=fn_grad)
amsgrad_ = amsgrad(fn = holder_fn, fn_grad=fn_grad)
nag_ = nag(fn = holder_fn, fn_grad=fn_grad)
RMSprop_ = RMSprop(fn = holder_fn, fn_grad=fn_grad)


# Obtain the path taken by plain gradient descent 
x_path, y_path, _ = plain_gradient_descent_.run(x_init=x_init,y_init=y_init, n_iter=n_iter, lr=lr, tol=1e-5)
pv_path = np.vstack((x_path, y_path))
# Obtain the path taken by momentum gradient descent 
x_path, y_path, _ = momentum_.run(x_init=x_init,y_init=y_init, n_iter=n_iter,lr=lr, beta=0.9, tol= 1e-5)
momentum_path = np.vstack((x_path, y_path))
# Obtain the path taken by adam 
x_path, y_path, _ = adam_.run(x_init=x_init,y_init=y_init, n_iter= n_iter, lr=lr, beta_1=0.9, beta_2=0.99, tol=1e-5, epsilon=1e-8)
adam_path = np.vstack((x_path, y_path))
# Obtain the path taken by adamax 
x_path, y_path, _ = adamax_.run(x_init=x_init, y_init=y_init, n_iter= n_iter, lr=lr, beta_1 = .9, beta_2 = .99, tol= 1e-5, epsilon = 1e-8)
adamax_path = np.vstack((x_path, y_path))
# Obtain the path taken by nadam 
x_path, y_path, _ = nadam_.run(x_init=x_init,y_init=y_init, n_iter=n_iter,lr=lr, beta_1 =.9, beta_2 =.99, tol=1e-5, epsilon=1e-8)
nadam_path = np.vstack((x_path, y_path))
# Obtain the path taken by amsgrad
x_path, y_path, _ = amsgrad_.run(x_init=x_init,y_init=y_init, n_iter=n_iter,lr=lr, beta_1 = .9, beta_2 = .99, tol= 1e-5, epsilon = 1e-8)
amsgrad_path = np.vstack((x_path, y_path))
# Obtain the path taken by nag 
x_path, y_path, _ = nag_.run(x_init=x_init,y_init=y_init, n_iter=n_iter,lr=lr, beta = .9, tol= 1e-5)
nag_path = np.vstack((x_path, y_path))
# Obtain the path taken by RMSprop
x_path, y_path, _ = RMSprop_.run(x_init=x_init,y_init=y_init, n_iter=n_iter,lr=lr, beta = .9, tol= 1e-5, epsilon = 1e-8)
rmsprop_path = np.vstack((x_path, y_path))





methods = [
    "Plain Vanilla",
    "Momentum",
    "Adam",
    "AdaMax",
    "Nadam",
    "AMSGrad",
    "Nesterov accelerated gradient",
    "RMSprop"
]

pv_df = pd.DataFrame(pv_path)
pv_df.rename(index = {0: "Plain Vanilla", 1:"Plain Vanilla"}, inplace = True) 
momen_df = pd.DataFrame(momentum_path)
momen_df.rename(index = {0: "Momentum", 1:"Momentum"}, inplace = True) 
adam_df = pd.DataFrame(adam_path)
adam_df.rename(index = {0: "Adam", 1:"Adam"}, inplace = True) 
adamax_df = pd.DataFrame(adamax_path)
adamax_df.rename(index = {0: "AdaMax", 1:"AdaMax"}, inplace = True) 
nadam_df = pd.DataFrame(adamax_path)
nadam_df.rename(index = {0: "Nadam", 1:"Nadam"}, inplace = True) 
amsgrad_df = pd.DataFrame(adamax_path)
amsgrad_df.rename(index = {0: "AMSGrad", 1:"AMSGrad"}, inplace = True) 
rmsprop_df = pd.DataFrame(rmsprop_path)
rmsprop_df.rename(index = {0: "RMSprop", 1:"RMSprop"}, inplace = True)
nag_df = pd.DataFrame(nag_path)
nag_df.rename(index = {0: "Nesterov accelerated gradient", 1:"Nesterov accelerated gradient"}, inplace = True)


paths_ = pv_df
paths_ = paths_.append(momen_df)
paths_ = paths_.append(adam_df)
paths_ = paths_.append(adamax_df)
paths_ = paths_.append(nadam_df)
paths_ = paths_.append(amsgrad_df)
paths_ = paths_.append(rmsprop_df)
paths_ = paths_.append(nag_df)


# Draw path and loss path from individual methods
paths = [np.array(paths_.loc[method]) for method in methods]
zpaths = [holder_fn(*path) for path in paths]


# Plot figure
fig = plt.figure(figsize=(8, 5), dpi = 120)
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot_surface(x, y, z, rstride=1, cstride=1, 
                edgecolor='none', alpha=.5, cmap=plt.cm.cool)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.xaxis.set_tick_params(labelsize=4)
ax.yaxis.set_tick_params(labelsize=4)
ax.zaxis.set_tick_params(labelsize=4)

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

anim = TrajectoryAnimation3D(*paths, zpaths=zpaths, labels=methods, ax=ax)
ax.legend(loc="upper left", fontsize='xx-small')

plt.show()
