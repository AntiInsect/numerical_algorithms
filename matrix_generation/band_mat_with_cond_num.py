import numpy as np
import scipy.sparse as sp


def band_mat_with_cond_num(row, col, half_band_size, ampl, cond):
    diag_vals = np.random.randn(2 * half_band_size + 1) * ampl - ampl / 2
    diag_data = np.repeat(diag_vals[:, np.newaxis], min(row, col), axis=1)
    ret = sp.spdiags(diag_data,
                     np.arange(-half_band_size, half_band_size + 1, dtype=np.int64),
                     row, col).toarray()

    left_fr, right_fr = 0, 1e+7     # be careful with this number! wrong value can produce:
                                    # 1. Floating points overflows (need smaller)
                                    # 2. Unreachable condition number (need bigger)
    factor = (left_fr + right_fr) / 2

    factor_prev = 1
    cur_cond = -1
    while np.abs(cur_cond - cond) > 10**(-2):
        idx = np.arange(min(row, col) - half_band_size), np.arange(half_band_size, min(row, col))
        
        ret[idx] *= factor / factor_prev
        
        cur_cond = np.linalg.cond(ret)
        if cur_cond < cond:
            left_fr = factor
        else:
            right_fr = factor

        factor_prev = factor
        factor = (left_fr + right_fr) / 2
    return ret

if __name__ == "__main__":
    row, col, cond = 5, 4, 3
    half_band_size, ampl = 2, 50
    matrix = band_mat_with_cond_num(row, col, half_band_size, ampl, cond)
    print(matrix)
