import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def newton_method(previous_x, previous_y, D3_arrows_artists):
    current_x = previous_x - dx(previous_x, previous_y)/ddx(previous_x, previous_y)
    current_y = previous_y - dy(previous_x, previous_y)/ddy(previous_x, previous_y)

    # plot on 2D figure
    plt.arrow(previous_x, previous_y, current_x - previous_x, current_y - previous_y, head_width=0.3, width=0.001,
              color='blue', label="gradient")
    
    # plot on 3D figure
    a = Arrow3D([previous_x, current_x], [previous_y, current_y],
                [func(previous_x, previous_y), func(current_x, current_y)], mutation_scale=10,
                lw=1, arrowstyle="-|>", color="blue")
    D3_arrows_artists.append(a)

    if abs(func(current_x, current_y) - func(previous_x, previous_y)) > 10**(-10):
        newton_method(current_x, current_y, D3_arrows_artists)
    else:
        print(str([current_x, current_y]) + " <----- newtone method")
        return [current_x, current_y]


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


if __name__ == "__main__":

    func = lambda X, Y : 5*X**2 + 4*X*Y + 2*Y**2 + 6*X + 7*Y
    dx   = lambda x, y : 10*x + 4*y + 6
    dy   = lambda x, y : 4*x + 4*y + 7
    ddx  = lambda x, y : 10
    ddy  = lambda x, y : 4


    STARTING_X = -10
    STARTING_Y = -10

    RANGE_START = -10
    RANGE_END = 10

    X = np.array([i for i in np.linspace(RANGE_START, RANGE_END, 1000)])
    Y = np.array([i for i in np.linspace(RANGE_START, RANGE_END, 1000)])
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

    colors_rec = ['blue']
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=colors_rec[0])]
    plt.legend(proxy, ["newtone method"])

    D3_arrows_artists = []
    newton_method(STARTING_X, STARTING_Y, D3_arrows_artists)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for artist in D3_arrows_artists:
        ax.add_artist(artist)

    surf = ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.5, linewidth=0)
    plt.colorbar(surf)
    plt.legend(proxy, ["newton method"])
    plt.show()
