import numpy as np


def mat_overdetermined(row, col, ampl):
    return np.random.randn(row, col) * ampl - ampl / 2

if __name__ == "__main__":
    row, col = 5, 4
    ampl = 50
    matrix = mat_overdetermined(row, col, ampl)
    print(matrix)
