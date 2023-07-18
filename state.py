from typing import Any
import numpy as np


class Index:
    """
    The following defines how the state vector is arranged.
    State vector X (14, ):
        position r (3)
        velocity v (3)
        mass m (1)
        position costate lr (6)
        velocity costate lv (3)
        mass costate lm (1)  
    """
    r = rp = range(0, 3)
    v = vp = range(3, 6)
    m = mp = range(6, 7)
    lr = lrp = range(7, 10)
    lv = lvp = range(10, 13)
    lm = lmp = range(13, 14)

    # costate vector range
    l = range(7, 14)
    
    # position-velocity vector range
    rv = range(0, 6)

    @staticmethod
    def i(row, col):
        return np.ix_(row, col)
        

class Jacobian:
    def __init__(self, arr):
        self.jac_arr = arr
        self.n_row, self.n_col = arr.shape

    def __setitem__(self, key, val):
        row, col = key
        sub_mat = val
        self.set_item(row, col, sub_mat)

    def set_item(self, row, col, sub_mat, check_zeros=True):
        sub_mat_range = np.ix_(row, col)
        sub_mat = np.atleast_1d(sub_mat)
        target_block = self.jac_arr[sub_mat_range]
        shape = target_block.shape
        if check_zeros:
            assert np.all(target_block == 0.0)
        self.jac_arr[sub_mat_range] = sub_mat.reshape(shape)
