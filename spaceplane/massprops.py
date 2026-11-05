"""Mass properties.

Inertia is linearly interpolated between a full-propellant tensor and a
zero-propellant tensor. That is a schedule through two measured states, not a
claim that propellant is a similar body. CG migration along body x is the
same kind of schedule. Product-of-inertia sign follows Stevens: the tensor is
diag(Ixx, Iyy, Izz) with -Ixz on the xz terms.

Variable-mass Euler equation (Thomson):
    I omega_dot + I_dot omega = M - omega x (I omega)
Exhaust momentum is already inside the thrust force/moment. Gravity-gradient
torque is added separately. CG is assumed to stay on the body x axis.
"""

from __future__ import annotations

import numpy as np


def inert_mass(cfg: dict) -> float:
    m = cfg["mass"]
    return float(m["dry_kg"] + m.get("payload_kg", 0.0))


def propellant_capacity(cfg: dict) -> float:
    m = cfg["mass"]
    return float(m["fuel_airbreathing_kg"] + m["rocket_propellant_kg"])


def total_mass(cfg: dict, m_ab: float, m_rocket: float) -> float:
    return inert_mass(cfg) + float(m_ab) + float(m_rocket)


def propellant_fraction(cfg: dict, m_ab: float, m_rocket: float) -> float:
    cap = propellant_capacity(cfg)
    if cap <= 0.0:
        return 0.0
    return float(np.clip((m_ab + m_rocket) / cap, 0.0, 1.0))


def cg_from_nose(cfg: dict, m_ab: float, m_rocket: float) -> float:
    f = propellant_fraction(cfg, m_ab, m_rocket)
    cg = cfg["cg_from_nose_m"]
    return float(cg["dry"] + f * (cg["full"] - cg["dry"]))


def _tensor(block: dict) -> np.ndarray:
    Ixx = float(block["Ixx"])
    Iyy = float(block["Iyy"])
    Izz = float(block["Izz"])
    Ixz = float(block.get("Ixz", 0.0))
    return np.array(
        [
            [Ixx, 0.0, -Ixz],
            [0.0, Iyy, 0.0],
            [-Ixz, 0.0, Izz],
        ],
        dtype=float,
    )


def inertia_tensor(cfg: dict, m_ab: float, m_rocket: float) -> np.ndarray:
    f = propellant_fraction(cfg, m_ab, m_rocket)
    I_dry = _tensor(cfg["inertia_dry"])
    I_full = _tensor(cfg["inertia_full"])
    return I_dry + f * (I_full - I_dry)


def inertia_rate(cfg: dict, mdot_ab: float, mdot_rocket: float) -> np.ndarray:
    """mdot_* are positive consumption rates (kg/s). Fraction then falls."""
    cap = propellant_capacity(cfg)
    if cap <= 0.0:
        return np.zeros((3, 3))
    df_dt = -(mdot_ab + mdot_rocket) / cap
    I_dry = _tensor(cfg["inertia_dry"])
    I_full = _tensor(cfg["inertia_full"])
    return df_dt * (I_full - I_dry)


def cylinder_inertia(mass_kg: float, radius_m: float, length_m: float) -> dict:
    """Solid circular cylinder about its CG (Hibbeler / standard rigid body).

    x is the long axis in the usual derivation; here Ixx is the roll axis
    (along the length) and Iyy = Izz are the transverse axes. This is only a
    placeholder when a mass-properties report does not exist.
    """
    Ixx = 0.5 * mass_kg * radius_m * radius_m
    Iyy = mass_kg * (3.0 * radius_m * radius_m + length_m * length_m) / 12.0
    return {"Ixx": Ixx, "Iyy": Iyy, "Izz": Iyy, "Ixz": 0.0}
