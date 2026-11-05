from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Command:
    """Controls. Angles in radians. Throttle in [0, 1]."""

    elevator_rad: float = 0.0
    throttle: float = 0.0
    mode: str = "none"
    alpha_rad: float | None = None
    bank_rad: float = 0.0
    # True: 3DOF sets the body triad from alpha. False: 6DOF uses the quaternion.
    kinematic_alpha: bool = True


@dataclass
class AeroResult:
    force_body_N: np.ndarray
    moment_body_Nm: np.ndarray
    CL: float
    CD: float
    CY: float
    Cm_noseup: float
    Cl_roll: float
    Cn_yaw: float
    alpha_rad: float
    beta_rad: float
    mach: float
    dynamic_pressure_Pa: float
    reynolds: float
    Cf: float
    L_N: float
    D_N: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class PropulsionResult:
    thrust_N: float
    mdot_fuel_kg_s: float
    mdot_air_kg_s: float
    isp_s: float
    tsfc_kg_Ns: float
    mode: str
    mach_exit: float
    stations: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConstraintReport:
    q_Pa: float
    qdot_W_m2: float
    n_total: float
    n_z_up: float
    alpha_deg: float
    beta_deg: float
    t_rad_K: float
    violated: list[str] = field(default_factory=list)
