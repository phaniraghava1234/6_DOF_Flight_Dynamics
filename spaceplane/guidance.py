"""Guidance laws.

Energy climb: at the current (h, V) pick the angle of attack that maximizes
Ps = (T cos alpha - D) V / W inside the path constraints (Rutowski 1954;
Bryson, Desai, Hoffman 1969). The search is algebraic, not a second integration.
Re-planned on a time bucket so RK4 substeps do not chatter.

Gravity turn: a pull-up at fixed alpha until gamma reaches a switch value, then
alpha = 0 so gravity bends the flight path (standard launch steering; the
vacuum optimal form is Lawden's linear tangent, offered as an option).

Parameter optimization of this steering law is direct shooting (Betts 1998),
implemented in mission.py. It is not a pseudospectral transcription.
"""

from __future__ import annotations

import math

from .atmosphere import us76
from .performance import ps_sweep
from .types import Command


def select_mode(mach: float, rho: float, m_ab: float, m_rocket: float, prop: dict) -> str:
    if m_ab > 0.0 and rho > 1e-3 and mach <= float(prop["mach_turbo_max"]):
        return "turbojet"
    if m_ab > 0.0 and rho > 1e-3 and mach <= float(prop["mach_ram_max"]):
        return "ramjet"
    if m_ab > 0.0 and rho > 1e-4 and mach <= float(prop["mach_scram_max"]):
        return "scramjet"
    if m_rocket > 0.0:
        return "rocket"
    return "none"


class EnergyClimb:
    def __init__(self, vehicle, bucket_s: float = 5.0):
        self.vehicle = vehicle
        self.bucket = bucket_s
        self._key = None
        self._cmd = Command(0.0, 1.0, "turbojet", 0.0, 0.0, True)

    def __call__(self, t: float, snap: dict) -> Command:
        key = int(math.floor(t / self.bucket))
        if key == self._key:
            # Keep the plan but allow the mode selector to see an empty tank.
            cmd = self._cmd
            return Command(cmd.elevator_rad, cmd.throttle, cmd.mode, cmd.alpha_rad, cmd.bank_rad, True)
        self._key = key
        prop = self.vehicle.cfg["propulsion"]
        mode = select_mode(snap["mach"], snap["rho"], snap["m_ab"], snap["m_rocket"], prop)
        if mode == "rocket" or mode == "none":
            self._cmd = Command(0.0, 1.0 if mode == "rocket" else 0.0, mode, 0.0, 0.0, True)
            return self._cmd
        best, _ = ps_sweep(
            self.vehicle,
            snap["h"],
            max(snap["V_air"], 1.0),
            mode,
            1.0,
            snap["m_ab"],
            snap["m_rocket"],
            n=7,
        )
        alpha = 0.0 if best is None else best["alpha"]
        self._cmd = Command(0.0, 1.0, mode, alpha, 0.0, True)
        return self._cmd


class GravityTurn:
    """Pull up at alpha_pull until gamma_air >= gamma_switch, then alpha = 0.

    Optional circularization: when inertial speed exceeds `circ_start_fraction`
    of local circular speed and altitude is inside the band, hold alpha_circ.
    """

    def __init__(self, alpha_pull: float, gamma_switch: float, alpha_circ: float, h_target: float, circ_fraction: float = 0.9):
        self.alpha_pull = alpha_pull
        self.gamma_switch = gamma_switch
        self.alpha_circ = alpha_circ
        self.h_target = h_target
        self.circ_fraction = circ_fraction

    def __call__(self, t: float, snap: dict) -> Command:
        from .gravity import circular_speed

        mode = "rocket" if snap["m_rocket"] > 0.0 else "none"
        throttle = 1.0 if mode == "rocket" else 0.0
        if snap["m_ab"] > 0.0 and snap["mach"] <= float("inf"):
            pass
        gamma = snap["gamma_air"]
        r = float(np_norm(snap["r"]))
        v_circ = circular_speed(r)
        if snap["h"] > 0.8 * self.h_target and snap["V_iner"] if "V_iner" in snap else 0:
            pass
        V_iner = float(np_norm(snap["v"]))
        if snap["h"] > 0.75 * self.h_target and V_iner > self.circ_fraction * v_circ:
            alpha = self.alpha_circ
        elif gamma < self.gamma_switch:
            alpha = self.alpha_pull
        else:
            alpha = 0.0
        return Command(0.0, throttle, mode, alpha, 0.0, True)


class LinearTangent:
    """Lawden linear-tangent pitch of the velocity-relative thrust angle.

    theta(t) = atan(a + b t) is the inertial pitch of the body x axis above
    the local horizontal. alpha = theta - gamma. Vacuum-optimal structure;
    using it inside the atmosphere is an engineering option, not a claim of
    optimality with aero forces.
    """

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __call__(self, t: float, snap: dict) -> Command:
        theta = math.atan(self.a + self.b * t)
        alpha = theta - snap["gamma_inertial"]
        mode = "rocket" if snap["m_rocket"] > 0 else "none"
        return Command(0.0, 1.0 if mode == "rocket" else 0.0, mode, alpha, 0.0, True)


def np_norm(x) -> float:
    import numpy as np

    return float(np.linalg.norm(x))


class AscentGuidance:
    """Airbreathing energy climb, then the selected rocket law."""

    def __init__(self, vehicle, rocket_law):
        self.vehicle = vehicle
        self.energy = EnergyClimb(vehicle)
        self.rocket = rocket_law

    def __call__(self, t: float, snap: dict) -> Command:
        prop = self.vehicle.cfg["propulsion"]
        mode = select_mode(snap["mach"], snap["rho"], snap["m_ab"], snap["m_rocket"], prop)
        if mode in ("turbojet", "ramjet", "scramjet"):
            return self.energy(t, snap)
        if mode == "rocket":
            cmd = self.rocket(t, snap)
            return Command(cmd.elevator_rad, cmd.throttle, "rocket", cmd.alpha_rad, 0.0, True)
        return Command(0.0, 0.0, "none", 0.0, 0.0, True)
