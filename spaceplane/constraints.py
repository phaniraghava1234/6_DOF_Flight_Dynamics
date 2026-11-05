"""Path constraints.

Dynamic pressure: q = 0.5 rho V^2.
Load factor: |F_nongrav| / (m g0), and nz_up = -Fz / (m g0) in the z-down frame.
Stagnation heating, Sutton and Graves, NASA TR R-376:
    q_dot = K * sqrt(rho / Rn) * V^3
K is a vehicle/atmosphere input. Secondary sources quote more than one SI value
for Earth air; the default in the example JSON is 1.83e-4 and must be confirmed
against TR R-376 before it is used for thermal design.
Uncoupled radiation balance (insulated backside, no hot-wall correction):
    sigma * epsilon * T^4 = q_dot
This is an upper-bound wall temperature, not a TPS sizing result.
"""

from __future__ import annotations

import math

from .constants import G0, SIGMA_SB
from .types import ConstraintReport


def stagnation_heat_flux(rho: float, V: float, Rn: float, K: float) -> float:
    if rho <= 0.0 or V <= 0.0 or Rn <= 0.0 or K <= 0.0:
        return 0.0
    return float(K * math.sqrt(rho / Rn) * V ** 3)


def radiation_temperature(qdot: float, emissivity: float) -> float:
    if qdot <= 0.0 or emissivity <= 0.0:
        return 0.0
    return float((qdot / (emissivity * SIGMA_SB)) ** 0.25)


def evaluate(
    limits: dict,
    *,
    rho: float,
    V_air: float,
    force_body,
    mass: float,
    alpha_rad: float,
    beta_rad: float,
) -> ConstraintReport:
    import numpy as np

    q = 0.5 * rho * V_air * V_air
    Rn = float(limits.get("nose_radius_m", limits.get("Rn_m", 0.1)))
    # nose radius may live on geometry; caller passes it inside limits copy
    K = float(limits["sutton_graves_k"])
    qdot = stagnation_heat_flux(rho, V_air, Rn, K)
    eps = float(limits.get("emissivity", 0.8))
    Trad = radiation_temperature(qdot, eps)
    F = np.asarray(force_body, dtype=float)
    n_tot = float(np.linalg.norm(F) / (mass * G0)) if mass > 0 else 0.0
    n_z_up = float(-F[2] / (mass * G0)) if mass > 0 else 0.0
    a_deg = math.degrees(alpha_rad)
    b_deg = math.degrees(beta_rad)
    bad = []
    if q > float(limits["q_max_pa"]):
        bad.append("dynamic_pressure")
    if qdot > float(limits["qdot_max_w_m2"]):
        bad.append("stagnation_heating")
    if n_tot > float(limits["n_max"]):
        bad.append("load_factor")
    if abs(a_deg) > float(limits["alpha_max_deg"]):
        bad.append("alpha")
    if abs(b_deg) > float(limits["beta_max_deg"]):
        bad.append("beta")
    if Trad > float(limits.get("wall_temperature_limit_k", 1e9)):
        bad.append("wall_temperature")
    return ConstraintReport(q, qdot, n_tot, n_z_up, a_deg, b_deg, Trad, bad)


def apply_governor(alpha: float, throttle: float, report: ConstraintReport, limits: dict) -> tuple[float, float, list[str]]:
    """Envelope protection. This is a control law, not a change to the EOM.

    Proportional pullback on the command when a path constraint is active.
    Gains are deliberately simple and are not claimed to be an optimal filter.
    """
    notes = []
    a = alpha
    th = throttle
    a_max = math.radians(float(limits["alpha_max_deg"]))
    if "dynamic_pressure" in report.violated or "stagnation_heating" in report.violated:
        a *= 0.5
        th *= 0.7
        notes.append("governor: reduced alpha and throttle for q or heating")
    if "load_factor" in report.violated:
        a *= 0.5
        notes.append("governor: reduced alpha for load factor")
    if "wall_temperature" in report.violated:
        th *= 0.7
        a *= 0.7
        notes.append("governor: reduced command for radiated wall temperature")
    a = float(np_clip(a, -a_max, a_max))
    th = float(max(0.0, min(1.0, th)))
    return a, th, notes


def np_clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x
