import numpy as np
import scipy.sparse as sp


def band_mx(row, col, half_band_size, ampl):
    diag_vals = np.random.randn(2 * half_band_size + 1) * ampl - ampl / 2
    diag_data = np.repeat(diag_vals[:, np.newaxis], min(row, col), axis=1)
    mx = sp.spdiags(diag_data, 
                    np.arange(-half_band_size, half_band_size + 1, dtype=np.int64),
                    row, col)
    return mx, mx.toarray()

if __name__ == "__main__":
    row, col = 7, 6
    half_band_size, ampl = 1, 50
    _, matrix = band_mx(row, col, half_band_size, ampl)
    print(matrix)
