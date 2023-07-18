import numpy as np
from astropy.time import Time, TimeDelta

DAY2SEC = 86400
DEG2RAD = np.pi / 180.0  # Degrees to radians
RAD2DEG = 180.0 / np.pi  # Radians to degrees

T_J2000 = Time("J2000", scale="tdb")

def mjd2sec(mjd):
    T = Time(mjd, format="mjd", scale="tdb")
    return (T - T_J2000).sec

def sec2mjd(sec):
    T = T_J2000 + TimeDelta(sec, format="sec", scale="tdb")
    return T.mjd

DU = 1.49597870691e8  # Distance Unit [km]
TU = 365.25 * DAY2SEC / (2 * np.pi)  # Time Unit [sec]
MaU = 4000  # Mass Unit [kg]