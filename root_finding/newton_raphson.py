import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def newton_raphson(f, f_pr, x0):
    xnew = xold = x0
    dx = 1000.
    iters = 0
    max_iters = 1000
    
    while dx > 10**(-5):
        iters += 1
        if iters > max_iters:
            print("Exceeded {0:3d} iterations without converging".format(max_iters))
            return 0
            
        xnew = xold - (1./f_pr(xold)) * f(xold)
        dx = np.abs(xnew-xold)
        xold = xnew

    return xnew


def method_vis(f, f_pr, x0, ax):
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
        xnew = xold - (1./f_pr(xold)) * f(xold)
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
  

def method_animate(f, f_pr, x0, ax, fig):
    ax.plot(x0, 0, 'go', label='x0')
    dx = 1000.
    iters = 0
    max_iters = 1000
    xnew = xold = x0
    
    # initialize data arrays 
    data = [[x0],[0.]]
    while dx > 10**(-5):
        iters += 1
        if iters>1000:
            print("Exceeded {0:3d} iterations without converging".format(max_iters))
            return 0

        xnew = xold - (1./f_pr(xold)) * f(xold)
        dx = np.abs(xnew - xold)
        
        data[0].append(xold)
        data[0].append(xnew)
        data[1].append(f(xold))
        data[1].append(0.)
    
        xold = xnew
     
        def update_line(num, data, line):
            line.set_data(data[..., :num])
            return line
    
        data = np.array(data)
        set_axis_lims(ax, data, f)
        line = plt.plot([],[],'r-')
        n = len(data[0,:])
        line_ani = animation.FuncAnimation(fig, update_line, n, 
                                           fargs=(data,line),
                                           interval=500,
                                           blit=True)

        ax.plot(xnew, 0, 'ro')
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

    
    fn = lambda x: x**3 - x + np.sqrt(2.)/2.
    fn_der = lambda x: 3*x**2 -1
    x_init = 11.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid('on')

    try:
        # x = newton_raphson(fn, fn_der, x_init)
        x = method_vis(fn, fn_der, x_init, ax)
        # x = method_animate(fn, fn_der, x_init, ax, fig)
        print("final value:", x)
    except ZeroDivisionError:
        print("Divided by zero.")
        print("Try again with different initial guess.")


