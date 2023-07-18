from numpy import tanh
from numba import njit


@njit
def throttle_tanh(sf, rho=1.0):
    """
    Throttle function approximated by hyperbolic tangent function
    as a smooth approximation.

    Parameters
    ----------
    sf : float
        Switching function.

    References
    ----------
    [1] Taheri E, Junkins J L. Generic smoothing for optimal bang-off-bang 
    spacecraft maneuvers[J]. Journal of Guidance, Control, and Dynamics,
    2018, 41(11): 2470-2475.
    """
    throttle = 0.5 * (1 + tanh(sf / rho))
    return throttle


@njit
def throttle_tanh_deriv(sf, rho=1.0):
    # compute derivative of tanh throttle function w.r.t. the switching funciton
    d_throttle_d_sf = 0.5 * (1 - tanh(sf / rho) ** 2) / rho
    return d_throttle_d_sf
