import numpy as np
from numpy.linalg import norm
from numba import njit

# For all function the gravitational parameter mu is assumed to be 1.

@njit
def g(r):
    """Calculate the gravitational acceleration vector.

    Parameters
    ----------
    r : np.ndarray, shape (3, )
        Position vector

    Returns
    -------
    g : np.ndarray, shape (3, )
        Gravitational acceleration vector
    """
    return -r / norm(r) ** 3

@njit
def dgdr(r):
    """Calculate the gradient of the gravitational acceleration vector.

    Parameters
    ----------
    r : np.ndarray, shape (3, )
        Position vector

    Returns
    -------
    G : np.ndarray, shape (3, 3)
        Gradient of the gravitational acceleration vector
    """
    G = 3 * np.outer(r, r) / norm(r) ** 5 - np.eye(3) / norm(r) ** 3
    return G


@njit
def delta(i, j):
    """Kronecker delta function.

    Parameters
    ----------
    i : int
        First index
    j : int
        Second index

    Returns
    -------
    delta : int
        1 if i == j, 0 otherwise
    """
    return 1 if i == j else 0


@njit
def d2gdr2(r):
    """Calculate the Hessian (d^2g/dr^2) of the gravitational acceleration vector.

    Parameters
    ----------
    r : np.ndarray, shape (3, )
        Position vector

    Returns
    -------
    H : np.ndarray, shape (3, 3, 3)
        Hessian of the gravitational acceleration vector
    """
    H = np.zeros((3, 3, 3))
    r_ = norm(r)
    for i in range(3):
        for j in range(3):
            dij = delta(i, j)
            for k in range(3):
                dik = delta(i, k)
                djk = delta(k, j)
                H[i, j, k] = (
                    3 / r_ ** 5 * (dij * r[k] + dik * r[j] + djk * r[i])
                    - 15 / r_ ** 7 * r[i] * r[j] * r[k]
                )
    return H