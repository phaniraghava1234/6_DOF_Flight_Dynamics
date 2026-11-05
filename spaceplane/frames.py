"""Coordinate frames and quaternions.

Scalar-first quaternion. v_inertial = R_body_to_inertial @ v_body.
Kinematics (Diebel; right-hand body rates):
    q_dot = 0.5 * Omega(omega) q
Positive body y rate is nose-up in the x-forward, z-down frame
(the rotation matrix R_y(+q) moves the nose to -z).
"""

from __future__ import annotations

import math

import numpy as np

from .constants import A_EARTH, OMEGA_EARTH


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def euler_omega_matrix(omega_body: np.ndarray) -> np.ndarray:
    p, q, r = (float(v) for v in omega_body)
    return np.array(
        [
            [0.0, -p, -q, -r],
            [p, 0.0, r, -q],
            [q, -r, 0.0, p],
            [r, q, -p, 0.0],
        ],
        dtype=float,
    )


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    return 0.5 * euler_omega_matrix(omega_body) @ q


def dcm_body_to_inertial(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = (float(v) for v in q)
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=float,
    )


def dcm_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q0 = 0.25 * s
        q1 = (R[2, 1] - R[1, 2]) / s
        q2 = (R[0, 2] - R[2, 0]) / s
        q3 = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q0 = (R[2, 1] - R[1, 2]) / s
        q1 = 0.25 * s
        q2 = (R[0, 1] + R[1, 0]) / s
        q3 = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q0 = (R[0, 2] - R[2, 0]) / s
        q1 = (R[0, 1] + R[1, 0]) / s
        q2 = 0.25 * s
        q3 = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q0 = (R[1, 0] - R[0, 1]) / s
        q1 = (R[0, 2] + R[2, 0]) / s
        q2 = (R[1, 2] + R[2, 1]) / s
        q3 = 0.25 * s
    q = np.array([q0, q1, q2, q3], dtype=float)
    return q / np.linalg.norm(q)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-15:
        raise ValueError("quaternion norm collapsed")
    return q / n


def spherical_position(lat_rad: float, lon_rad: float, h_m: float, radius: float = A_EARTH) -> np.ndarray:
    cl, sl = math.cos(lat_rad), math.sin(lat_rad)
    co, so = math.cos(lon_rad), math.sin(lon_rad)
    r = radius + h_m
    return np.array([r * cl * co, r * cl * so, r * sl], dtype=float)


def ned_to_ecef_dcm(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Columns are north, east, down expressed in ECEF."""
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    so, co = math.sin(lon_rad), math.cos(lon_rad)
    return np.array(
        [
            [-sl * co, -so, -cl * co],
            [-sl * so, co, -cl * so],
            [cl, 0.0, -sl],
        ],
        dtype=float,
    )


def eci_to_ecef(r_eci: np.ndarray, t_s: float, omega: float = OMEGA_EARTH) -> np.ndarray:
    th = -omega * t_s
    c, s = math.cos(th), math.sin(th)
    x, y, z = (float(v) for v in r_eci)
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float)


def ecef_spherical_lat_lon(r_ecef: np.ndarray) -> tuple[float, float]:
    x, y, z = (float(v) for v in r_ecef)
    rn = math.sqrt(x * x + y * y + z * z)
    lat = math.asin(clamp(z / rn, -1.0, 1.0))
    lon = math.atan2(y, x)
    return lat, lon


def air_velocity_eci(v_eci: np.ndarray, r_eci: np.ndarray, omega: float = OMEGA_EARTH) -> np.ndarray:
    w = np.array([0.0, 0.0, omega], dtype=float)
    return np.asarray(v_eci, dtype=float) - np.cross(w, r_eci)


def lift_up_direction(v_hat: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to v, toward local up (radial)."""
    u = r_hat - float(np.dot(r_hat, v_hat)) * v_hat
    n = float(np.linalg.norm(u))
    if n < 1e-10:
        # Nearly radial flight: pick a horizontal axis from the inertial z.
        tmp = np.array([0.0, 0.0, 1.0]) if abs(float(v_hat[2])) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = tmp - float(np.dot(tmp, v_hat)) * v_hat
        n = float(np.linalg.norm(u))
    return u / n


def body_triad(v_hat: np.ndarray, r_hat: np.ndarray, alpha_rad: float, bank_rad: float = 0.0) -> np.ndarray:
    """Columns of R_body_to_inertial.

    Zero bank, alpha = 0: x along air-relative velocity, z down, y to the right.
    Positive alpha: nose toward local up. Positive bank: right wing down
    (body y rotates toward body z).
    """
    u_L = lift_up_direction(v_hat, r_hat)
    ca, sa = math.cos(alpha_rad), math.sin(alpha_rad)
    x0 = ca * v_hat + sa * u_L
    z0 = sa * v_hat - ca * u_L
    y0 = np.cross(z0, x0)
    x0 = x0 / np.linalg.norm(x0)
    y0 = y0 / np.linalg.norm(y0)
    z0 = z0 / np.linalg.norm(z0)
    c, s = math.cos(bank_rad), math.sin(bank_rad)
    x = x0
    y = c * y0 + s * z0
    z = -s * y0 + c * z0
    R = np.column_stack((x, y, z))
    return R


def alpha_beta_from_body_velocity(v_body: np.ndarray) -> tuple[float, float, float]:
    """Stevens and Lewis. u forward, v right, w down. Returns alpha, beta, speed."""
    u, v, w = (float(x) for x in v_body)
    speed = math.sqrt(u * u + v * v + w * w)
    if speed < 1e-6:
        return 0.0, 0.0, speed
    alpha = math.atan2(w, u)
    beta = math.asin(clamp(v / speed, -1.0, 1.0))
    return alpha, beta, speed


def decompose_aero_force(force_body: np.ndarray, alpha: float, beta: float) -> tuple[float, float, float]:
    """Return L (positive up), D (positive aft along relative wind), Y (positive right)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    v_hat = np.array([ca * cb, sb, sa * cb], dtype=float)
    D = -float(np.dot(force_body, v_hat))
    up = np.array([0.0, 0.0, -1.0])
    up_perp = up - float(np.dot(up, v_hat)) * v_hat
    n = float(np.linalg.norm(up_perp))
    if n < 1e-10:
        L = 0.0
        u_L = np.array([0.0, 0.0, -1.0])
    else:
        u_L = up_perp / n
        L = float(np.dot(force_body, u_L))
    u_Y = np.cross(v_hat, u_L)
    ny = float(np.linalg.norm(u_Y))
    Y = float(np.dot(force_body, u_Y / ny)) if ny > 1e-10 else 0.0
    return L, D, Y


def force_from_coefficients(L: float, D: float, Y: float, alpha: float, beta: float) -> np.ndarray:
    """Inverse of decompose at the same alpha, beta. Drag opposes velocity."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    v_hat = np.array([ca * cb, sb, sa * cb], dtype=float)
    up = np.array([0.0, 0.0, -1.0])
    up_perp = up - float(np.dot(up, v_hat)) * v_hat
    n = float(np.linalg.norm(up_perp))
    u_L = up_perp / n if n > 1e-10 else np.array([0.0, 0.0, -1.0])
    u_Y = np.cross(v_hat, u_L)
    ny = float(np.linalg.norm(u_Y))
    if ny > 1e-10:
        u_Y = u_Y / ny
    return -D * v_hat + L * u_L + Y * u_Y
