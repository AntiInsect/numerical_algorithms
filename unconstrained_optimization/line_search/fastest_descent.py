import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# https://github.com/lovelyscientist/func-optimization-methods
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


EPSILON = 0.000001
LAMDA = 0.9

STARTING_X = -10
STARTING_Y = -10

RANGE_START = -10
RANGE_END = 10
STEP_SIZE = 0.02
delta = EPSILON/1000
X = np.array([i for i in np.linspace(RANGE_START, RANGE_END, 1000)])
Y = np.array([i for i in np.linspace(RANGE_START, RANGE_END, 1000)])
D3_arrows_artists = []


def func(X, Y): return 5*X**2 + 4*X*Y + 2*Y**2 + 6*X + 7*Y
def dxy(x,y): return 4
def dx(x, y): return 10*x + 4*y + 6
def ddx(x,y): return 10
def ddy(x,y): return 4
def dy(x, y): return 4*x + 4*y + 7


def fastest_descent(previous_x, previous_y):
    delta_x = dx(previous_x, previous_y)
    delta_y = dy(previous_x, previous_y)

    a = 2*(5*(delta_x**2) + 2*(delta_y**2) + 4*delta_x*delta_y)
    b = 4*previous_y*delta_y + 6*delta_x + 7*delta_y + 10*previous_x*delta_x + 4*delta_x*previous_y + 4*delta_y*previous_x
    alpha = b/a

    current_x = previous_x - alpha * delta_x
    current_y = previous_y - alpha * delta_y

    plt.arrow(previous_x, previous_y, current_x - previous_x, current_y - previous_y, head_width=0.3, width=0.1,length_includes_head='True', color='green', label="gradient")
    a = Arrow3D([previous_x, current_x], [previous_y, current_y], [func(previous_x, previous_y), func(current_x, current_y)], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="green")
    D3_arrows_artists.append(a)

    if abs(func(current_x, current_y) - func(previous_x, previous_y)) > EPSILON:
        fastest_descent(current_x, current_y)
    else:
        print(str([current_x, current_y]) + " <----- fastest descent")
        return [current_x, current_y]


if __name__ == "__main__":
    xlist = np.linspace(RANGE_START, RANGE_END, 1000)
    ylist = np.linspace(RANGE_START, RANGE_END, 1000)
    X, Y = np.meshgrid(xlist, ylist)
    Z = func(X, Y)

    plt.figure()
    cp_nc = plt.contour(X, Y, Z, colours="black")
    plt.clabel(cp_nc, inline=True, fontsize=8)
    cp = plt.contourf(X, Y, Z, cmap='jet', alpha=0.5)
    plt.colorbar(cp)
    plt.title('Gradient Methods')
    plt.xlabel('X')
    plt.ylabel('Y')

    colors_rec = ['green']
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=colors_rec[0])]
    plt.legend(proxy, ["fastest descent"])
    fastest_descent(STARTING_X,STARTING_Y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for artist in D3_arrows_artists:
        ax.add_artist(artist)

    surf = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.5, linewidth=0)
    plt.colorbar(surf)
    plt.legend(proxy, ["fastest descent"])
    plt.show()
