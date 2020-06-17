import numpy as np

def hilbert(n):
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    return 1.0 / (i_grid + j_grid + 1)

if __name__ == "__main__":
    N = 10
    print(hilbert(N))

    import matplotlib.pyplot as plt
    plt.semilogy([np.linalg.cond(hilbert(i)) for i in np.arange(1, 50)])
    plt.show()
