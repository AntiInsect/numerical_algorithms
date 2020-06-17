import numpy as np
import matplotlib.pyplot as plt


def steffensen_method(f, x0, ax):
    xnew = xold = x0
    ax.plot(xold,0, 'go', label='x0')
    dx = 1000.
    iters = 0
    max_iters = 1000
    
    # initialize data arrays - useful for setting window size 
    
    data=[[x0],[0.]]
    
    while dx > 10**(-5):
        iters += 1
        if iters > 1000:
            print("Exceeded {0:3d} iterations without converging".format(max_iters))
            return 0
            
        # draw line from current x straight up to graph of f
        ax.plot((xold, xold), (0, f(xold)), 'k-')

        x1 = xold + f(xold)
        x2 = x1 + f(x1)

        xnew = x2 - (x2 - x1)**2 / (x2 - (2*x1) + xold)

        dx = np.abs(xnew - xold)
    
        #update data store
        data[0].append(xold)
        data[0].append(xnew)
        data[1].append(f(xold))
        data[1].append(0.)
	
        # draw line from graph of f to new estimate of root
        ax.plot((xold, xnew), (f(xold), 0), 'k-')
        ax.plot(xnew, 0, 'ko')
	
        xold = xnew
    
    ax.plot(xnew, 0, 'ro')
    data = np.array(data) 
    set_axis_lims(ax, data, f)
    plt.legend()
    plt.show()
    return xnew


def set_axis_lims(ax, data, f):
    # window overhang as percent of data width
    overhang = 0.5
    # number of points to use for plotting function
    num_pts = 1000. 
    
    xmin, xmax = np.min(data[0,:]), np.max(data[0,:])
    ymin, ymax = np.min(data[1,:]), np.max(data[1,:])
    delta_x, delta_y = xmax - xmin, ymax - ymin
  
    xlim_low = xmin - overhang * delta_x
    xlim_high = xmax + overhang * delta_x
    
    ylim_low = ymin - overhang * delta_y
    ylim_high = ymax + overhang * delta_y

    ax.set_xlim(xlim_low, xlim_high)
    ax.set_ylim(ylim_low, ylim_high)
  
    xvals = np.arange(xlim_low, xlim_high, delta_x/num_pts)
    yvals = f(xvals)
    ax.plot(xvals, yvals)

if __name__ == "__main__":

    fn = lambda x: (x*x)-(2*x)-56
    x_init = 0.7


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid('on')

    x = steffensen_method(fn, x_init, ax)
    print("final value:", x)
