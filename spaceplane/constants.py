"""Physical constants.

US76: NOAA/NASA/USAF, U.S. Standard Atmosphere, 1976.
WGS-84: NIMA TR8350.2 as used by Vallado, Fundamentals of Astrodynamics
and Applications. J2 is given to 8 digits so extra digits are not invented.
"""

# Standard gravity. Exact conventional value. Also the US76 hydrostatic g0.
G0 = 9.80665  # m/s^2

# COESA 1976
R_STAR = 8314.32  # J/(kmol K)
M_AIR = 28.9644  # kg/kmol
R_AIR = R_STAR / M_AIR  # J/(kg K)
P0_STD = 101325.0  # Pa
T0_STD = 288.15  # K
R0_GEOPOTENTIAL = 6356766.0  # m, US76 effective Earth radius
GAMMA_AIR = 1.4

# Sutherland viscosity, standard aeronautical constants:
# mu = 1.458e-6 * T^1.5 / (T + S), S = 110.4 K.
SUTHERLAND_B = 1.458e-6
SUTHERLAND_S = 110.4

# WGS-84 spherical model used here (ellipsoid geodesy is not implemented).
MU_EARTH = 3.986004418e14  # m^3/s^2
A_EARTH = 6378137.0  # m
J2_EARTH = 1.08262668e-3
OMEGA_EARTH = 7.292115e-5  # rad/s

SIGMA_SB = 5.670374419e-8  # W/(m^2 K^4), Stefan-Boltzmann
