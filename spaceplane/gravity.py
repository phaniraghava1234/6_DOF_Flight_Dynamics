"""Two-body plus J2 gravity.

Potential convention (Vallado): acceleration is the gradient of
    U = (mu/r) * [1 - J2 (Re/r)^2 (3 sin^2(phi) - 1)/2 ]
with sin(phi) = z/r. Analytic derivatives below. A unit test compares them
to a central difference of U so a typesetting error in the J2 terms fails
loudly. Gravity-gradient torque is in dynamics.py (Schaub and Junkins).
"""

from __future__ import annotations

import numpy as np

from .constants import A_EARTH, J2_EARTH, MU_EARTH


def potential(r_eci: np.ndarray, mu: float = MU_EARTH, j2: float = J2_EARTH, radius: float = A_EARTH) -> float:
    r = np.asarray(r_eci, dtype=float)
    rn = float(np.linalg.norm(r))
    s2 = (r[2] / rn) ** 2
    return (mu / rn) * (1.0 - j2 * (radius / rn) ** 2 * 0.5 * (3.0 * s2 - 1.0))


def acceleration(r_eci: np.ndarray, mu: float = MU_EARTH, j2: float = J2_EARTH, radius: float = A_EARTH) -> np.ndarray:
    """grad U, m/s^2. J2 = 0 reduces to -mu r / r^3."""
    x, y, z = (float(v) for v in np.asarray(r_eci, dtype=float))
    rn = math_norm(x, y, z)
    K = 0.5 * mu * j2 * radius * radius
    r2 = rn * rn
    r3 = r2 * rn
    r5 = r3 * r2
    r7 = r5 * r2
    ax = -mu * x / r3 + K * (15.0 * x * z * z / r7 - 3.0 * x / r5)
    ay = -mu * y / r3 + K * (15.0 * y * z * z / r7 - 3.0 * y / r5)
    az = -mu * z / r3 - 9.0 * K * z / r5 + 15.0 * K * z * z * z / r7
    return np.array([ax, ay, az], dtype=float)


def math_norm(x: float, y: float, z: float) -> float:
    return float(np.sqrt(x * x + y * y + z * z))


def circular_speed(r_m: float, mu: float = MU_EARTH) -> float:
    return float(np.sqrt(mu / r_m))


def orbital_elements(r_eci: np.ndarray, v_eci: np.ndarray, mu: float = MU_EARTH) -> dict:
    """Keplerian elements from r, v. J2 is ignored in the conversion (osculating two-body)."""
    r = np.asarray(r_eci, dtype=float)
    v = np.asarray(v_eci, dtype=float)
    rn = float(np.linalg.norm(r))
    vn = float(np.linalg.norm(v))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    e_vec = np.cross(v, h) / mu - r / rn
    e = float(np.linalg.norm(e_vec))
    energy = 0.5 * vn * vn - mu / rn
    if energy < 0.0:
        a = -mu / (2.0 * energy)
        rp = a * (1.0 - e)
        ra = a * (1.0 + e)
    else:
        a = float("inf")
        rp = float("nan")
        ra = float("inf")
    inc = float(np.arccos(np.clip(h[2] / hn, -1.0, 1.0))) if hn > 0 else 0.0
    return {
        "specific_energy_m2_s2": energy,
        "sma_m": a,
        "eccentricity": e,
        "inclination_rad": inc,
        "rp_m": rp,
        "ra_m": ra,
        "h_vec": h,
    }
