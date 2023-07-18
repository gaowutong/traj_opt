import numpy as np
from numpy import cos, sin, sqrt
from numpy.linalg import norm
from poliastro.core.elements import rv2coe, coe2mee
from numba import njit


def from_cartesian(cartesian, mu):
    r, v = cartesian[:3], cartesian[3:]
    coe = rv2coe(mu, r, v)
    mee = coe2mee(*coe)
    return mee


@njit
def to_cartesian(mee, mu):
    """Calculates position and velocity vector from modified equinoctial elements.

    Parameters
    ----------
    p: float
        Semi-latus rectum
    f: float
        Equinoctial parameter f
    g: float
        Equinoctial parameter g
    h: float
        Equinoctial parameter h
    k: float
        Equinoctial parameter k
    L: float
        Longitude
    mu: float
        Gravitational parameter

    Returns
    -------
    rv: numpy.ndarray
        Position and velocity vector.

    Note
    ----
    The definition of `r` and `v` is taken from
    https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf,
    Equation 3a and 3b.

    """
    p, f, g, h, k, L = mee
    w = 1 + f * np.cos(L) + g * np.sin(L)
    r = p / w
    s2 = 1 + h**2 + k**2
    alpha2 = h**2 - k**2

    rx = (r / s2) * (np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L))
    ry = (r / s2) * (np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L))
    rz = (2 * r / s2) * (h * np.sin(L) - k * np.cos(L))

    vx = (
        (-1 / s2)
        * (np.sqrt(mu / p))
        * (
            np.sin(L)
            + alpha2 * np.sin(L)
            - 2 * h * k * np.cos(L)
            + g
            - 2 * f * h * k
            + alpha2 * g
        )
    )
    vy = (
        (-1 / s2)
        * (np.sqrt(mu / p))
        * (
            -np.cos(L)
            + alpha2 * np.cos(L)
            + 2 * h * k * np.sin(L)
            - f
            + 2 * g * h * k
            + alpha2 * f
        )
    )
    vz = (
        (2 / s2)
        * (np.sqrt(mu / p))
        * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)
    )
    return np.array([rx, ry, rz, vx, vy, vz])


@njit
def AB(els, mu):
    p, f, g, h, k, l = els
    w = 1 + f * cos(l) + g * sin(l)
    c = 1 + h ** 2 + k ** 2
    z = h * sin(l) - k * cos(l)
    A = sqrt(mu / p) * np.array([
        0, 0, 0, 0, 0, w ** 2 / p
    ])
    B = sqrt(p / mu) * np.array([
        [0, 2 * p / w, 0],
        [sin(l), cos(l) + (f + cos(l)) / w, -z * g / w],
        [-cos(l), sin(l) + (g + sin(l)) / w, z * f / w],
        [0, 0, c * cos(l) / (2 * w)],
        [0, 0, c * sin(l) / (2 * w)],
        [0, 0, z / w]
    ])
    return A, B


@njit
def dAB(els, mu):
    p, f, g, h, k, l = els
    w = 1 + f * cos(l) + g * sin(l)
    dA = np.zeros((6, 6))
    dA[-1, :] = sqrt(mu / p) * np.array([
        -3 * w**2 / (2 * p**2),
        2 * w * cos(l) / p,
        2 * w * sin(l) / p,
        0,
        0,
        (-2 * f * sin(l) + 2 * g * cos(l)) * w / p,
    ])

    # derive by sympy
    x0 = sqrt(p / mu)
    x1 = cos(l)
    x2 = f * x1
    x3 = sin(l)
    x4 = g * x3
    x5 = x4 + 1
    x6 = x2 + x5
    x7 = 1 / x6
    x8 = x0 * x7
    x9 = x0 * x3
    x10 = 1 / p
    x11 = x10 / 2
    x12 = 2 * l
    x13 = cos(x12)
    x14 = x13 / 4
    x15 = sin(x12)
    x16 = x15 / 4
    x17 = x10 * x8
    x18 = h * x3
    x19 = -k * x1 + x18
    x20 = x11 * x8
    x21 = x0 * x1
    x22 = f / 2
    x23 = x21 * x7
    x24 = h**2 + k**2 + 1
    x25 = x10 * x24 / 4
    x26 = x7 * x9
    x27 = x6**2
    x28 = 1 / x27
    x29 = 2 * p
    x30 = x28 * x29
    x31 = g + x3
    x32 = x28 * x31
    x33 = k * x13
    x34 = -h * x15 + k + x33
    x35 = x0 * x28
    x36 = x35 / 2
    x37 = -x24 * x36
    x38 = -x16 * x24 * x35
    x39 = f + x1
    x40 = x28 * x39
    x41 = k * x1
    x42 = h * x13 - h + k * x15
    x43 = f * x3
    x44 = g * x1
    x45 = k * x3
    x46 = x43 - x44
    x47 = x35 * (f * h + g * k + h * x1 + x45)
    x48 = x24 * x36

    dB = np.array([
        [
            [0, 3 * x8, 0],
            [x11 * x9, x17 * (f * x14 + 3 * f / 4 + g * x16 + x1), -g * x19 * x20],
            [-x11 * x21, x17 * (f * x16 - g * x14 + 3 * g / 4 + x3), x17 * x19 * x22],
            [0, 0, x23 * x25],
            [0, 0, x25 * x26],
            [0, 0, x19 * x20]
        ],
        [
            [0, -x21 * x30, 0],
            [0, x32 * x9, -g * x34 * x36],
            [0, -x21 * x32, x19 * x35 * x5],
            [0, 0, x1**2 * x37],
            [0, 0, x38],
            [0, 0, x34 * x36]
        ],
        [
            [0, -x30 * x9, 0],
            [0, -x40 * x9, x35 * (-h * x15 * x22 + k * x22 - x18 + x22 * x33 + x41)],
            [0, x21 * x40, x22 * x35 * x42],
            [0, 0, x38],
            [0, 0, x3**2 * x37],
            [0, 0, x36 * x42]],
        [
            [0.0, 0, 0],
            [0, 0, -x4 * x8],
            [0, 0, x43 * x8],
            [0, 0, h * x23],
            [0, 0, x18 * x8],
            [0, 0, x26]
        ],
        [
            [0.0, 0, 0],
            [0, 0, x44 * x8],
            [0, 0, -x2 * x8],
            [0, 0, x41 * x8],
            [0, 0, x45 * x8],
            [0, 0, -x23]],
        [
            [0, x29 * x35 * x46, 0],
            [x21, x35 * (-x27 * x3 - x3 * x6 + x39 * x46), -g * x47],
            [x9, x35 * (x1 * x27 + x1 * x6 + x31 * x46), f * x47],
            [0, 0, -x31 * x48],
            [0, 0, x39 * x48],
            [0, 0, x47]
        ]
    ]).transpose(1, 2, 0)  # shape (6, 3, 6)
    return dA, dB