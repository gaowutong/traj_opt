import numpy as np
from numpy.linalg import norm
from numba.experimental import jitclass
from numba import float64, boolean

from gravity import g, dgdr, d2gdr2
from meoe import AB, dAB
from throttle_function import throttle_tanh, throttle_tanh_deriv
from state import Jacobian, Index as I


# @jitclass([
#     ('mu', float64),
#     ('thrust_max', float64),
#     ('Isp', float64),
#     ('g0', float64),
#     ('c', float64),
#     ('rho', float64),
#     ('variation', boolean)
# ])
class LowThrustTwoBody():
    def __init__(self, mu, thrust_max, Isp, g0, rho=1):
        """Dynamics of a low-thrust two-body problem.

        Parameters
        ----------
        mu : float
            Gravitational parameter of central body [DU^3/TU^2]
        thrust_max : float
            Maximum thrust [MaU*DU/TU^2]
        I_sp : float
            Specific impulse [TU]
        g0 : float
            Standard gravity [DU/TU^2]
        rho : float, default 1
            Continuation parameter [n.d.]
        """

        self.mu = mu
        self.thrust_max = thrust_max
        self.Isp = Isp
        self.g0 = g0
        self.c = Isp * g0  # exhaust velocity [DU/TU]
        self.rho = rho
        self.variation = False

    def time_derivative(self, t, x):
        """Calculate the time derivative of the system.

        Parameters
        ----------
        t : float
            Time [s]
        x : np.ndarray, shape (14 + 14*14, )
            State vector plus state transition matrix.
        Returns
        -------
        state_dot : np.ndarray
            Time derivative of the state vector and state transition matrix.        
        """
        Tmax = self.thrust_max
        c = self.c
        mu = self.mu

        # state vector X
        r = x[:3]
        v = x[3:6]
        m = x[6]
        lr = x[7:10]
        lv = x[10:13]
        lm = x[13]
        # assert m >= 0

        # switching function
        sf = c * norm(lv) / m + lm - 1

        # optimal throttle function (0-1) and thrust direction
        # delta = 1 for sf > 0 and 0 for sf < 0
        # but this is discontinuous, so we use tanh
        # as a smooth approximation (see Ref[1])
        delta = throttle_tanh(sf, self.rho)
        u = -lv / norm(lv)
        # define vector k
        k = u * delta

        # central gravitational and their derivatives
        grav_acc = mu * g(r)  # gravitational acceleration vector
        G = mu * dgdr(r)  # gravity gradient matrix

        thrust_acc = Tmax / m * k  # thrust acceleration vector

        # dXdt
        rp = v
        vp = grav_acc + thrust_acc
        mp = -Tmax / c * delta
        lrp = -G @ lv
        lvp = -lr
        lmp = Tmax / m ** 2 * k @ lv
        dXdt = np.array([
            *rp,
            *vp,
            mp,
            *lrp,
            *lvp,
            lmp
        ])
        if not self.variation:
            return dXdt

        # some intermediate derivatives
        # D for matrix, d for vector
        dG_dr = mu * d2gdr2(r)  # gravity hessian tensor

        # sf = c * norm(lv) / m + lm - 1
        dsf_dm = -c * norm(lv) / m ** 2
        dsf_dlm = 1
        dsf_dlv = c * lv / norm(lv) / m

        # derivatives
        ddelta_dsf = throttle_tanh_deriv(sf, self.rho)
        du_dlv = -np.eye(3) / norm(lv) + np.outer(lv, lv) / norm(lv) ** 3
        # k = u * delta
        dk_dm = u * ddelta_dsf * dsf_dm
        dk_dlv = np.outer(u, ddelta_dsf * dsf_dlv) + du_dlv * delta
        dk_dlm = u * ddelta_dsf * dsf_dlm

        Phi = x[14:].reshape(14, 14)
        dFdX = self.dFdX = Jacobian(np.zeros((14, 14)))
        dFdX[I.rp, I.v] = np.eye(3)  # drp/dv = I

        dFdX[I.vp, I.r] = G  # dvp/dr = dg/dr = G
        dFdX[I.vp, I.m] = Tmax / m * dk_dm - Tmax / m ** 2 * k
        dFdX[I.vp, I.lv] = Tmax / m * dk_dlv
        dFdX[I.vp, I.lm] = Tmax / m * dk_dlm

        # -Tmax / c * delta
        dmp_dsf = -Tmax / c * ddelta_dsf
        dFdX[I.mp, I.m] = dmp_dsf * dsf_dm  # dmp/dm
        dFdX[I.mp, I.lm] = dmp_dsf * dsf_dlm  # dmp/dlm
        dFdX[I.mp, I.lv] = dmp_dsf * dsf_dlv  # dmp/dlv

        dFdX[I.lr, I.r] = -np.einsum('ijk,j->ik', dG_dr, lv)  # dlr/dr
        dFdX[I.lr, I.lv] = -G  # dlr/dlv
        dFdX[I.lv, I.lr] = -np.eye(3)  # dlv/dlr

        # lmp = T_max / m ** 2 * k @ lv
        dFdX[I.lmp, I.m] = (
            -2 * Tmax / m ** 3 * k @ lv
            + Tmax / m ** 2 * dk_dm @ lv
        )
        dFdX[I.lmp, I.lv] = Tmax / m ** 2 * (k + lv @ dk_dlv)
        dFdX[I.lmp, I.lm] = Tmax / m ** 2 * (dk_dlm @ lv)

        dPhidt = dFdX.jac_arr @ Phi
        xp = np.concatenate((dXdt, dPhidt.ravel()))
        return xp


@jitclass([
    ('mu', float64),
    ('thrust_max', float64),
    ('Isp', float64),
    ('g0', float64),
    ('c', float64),
    ('rho', float64),
])
class LowThrustTwoBodyMEOE:
    def __init__(self, mu, thrust_max, Isp, g0, rho=1):
        self.mu = mu
        self.thrust_max = thrust_max
        self.Isp = Isp
        self.g0 = g0
        self.c = Isp * g0  # exhaust velocity [DU/TU]
        self.rho = rho

    def time_derivative(self, t, x):
        Tmax = self.thrust_max
        c = self.c
        mu = self.mu

        # state vector X
        el = x[:6]
        m = x[6]
        lambda_el = x[7:13]
        lambda_m = x[13]

        A, B = AB(el, mu)
        dA, dB = dAB(el, mu)

        # switching function
        p = -B.T @ lambda_el
        sf = c * norm(p) / m + lambda_m - 1
        # optimal throttle function (0-1) and thrust direction
        # delta = 1 for sf > 0 and 0 for sf < 0
        # but this is discontinuous, so we use tanh
        # as a smooth approximation (see Ref[1])
        delta = throttle_tanh(sf, self.rho)
        u = -p / norm(p)
        # define vector k
        k = u * delta

        thrust_acc = Tmax / m * k  # thrust acceleration vector

        # dXdt
        el_p = A + B @ thrust_acc
        mp = -Tmax / c * delta
        dB1 = dB[:, 0, :] * thrust_acc[0]
        dB2 = dB[:, 1, :] * thrust_acc[1]
        dB3 = dB[:, 2, :] * thrust_acc[2]
        d_elp_del = dA + dB1 + dB2 + dB3
        lambda_el_p = -d_elp_del.T @ lambda_el
        lambda_m_p = -Tmax / m ** 2 * norm(p) * delta
        dxdt = np.array([
            *el_p,
            mp,
            *lambda_el_p,
            lambda_m_p
        ])
        return dxdt