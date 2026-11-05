"""Equations of motion.

Translational EOM are written in an inertial frame (epoch ECI):
    r_dot = v
    v_dot = g(r) + R_b2i @ F_body / m
No Coriolis term is added. Aerodynamic velocity is v - omega_earth x r
(Vallado rotating-atmosphere construction).

Rotational EOM in body axes (Zipfel; Thomson variable inertia):
    I omega_dot = M - omega x (I omega) - I_dot omega
Gravity-gradient torque (Schaub and Junkins), optional:
    M_gg = (3 mu / r^3) r_hat_b x I r_hat_b
Quaternion kinematics: see frames.py. Renormalized after each full step.

3DOF uses the same forces with a kinematic body triad from alpha and bank
(Vinh's point-mass reduction: attitude is prescribed, moments are not integrated).
"""

from __future__ import annotations

import math

import numpy as np

from .atmosphere import us76
from .constraints import apply_governor, evaluate as eval_constraints
from .constants import A_EARTH, G0, MU_EARTH, OMEGA_EARTH
from .frames import (
    air_velocity_eci,
    alpha_beta_from_body_velocity,
    body_triad,
    dcm_body_to_inertial,
    normalize_quat,
    quat_derivative,
)
from .gravity import acceleration as grav_acc
from .massprops import cg_from_nose, inertia_rate, inertia_tensor, total_mass
from .types import Command


def rk4_step(rhs, t: float, y: np.ndarray, dt: float, quat_slice: slice | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    y2 = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if quat_slice is not None:
        y2[quat_slice] = normalize_quat(y2[quat_slice])
    return y2


def _limits_with_rn(vehicle_cfg: dict) -> dict:
    lim = dict(vehicle_cfg["limits"])
    lim["nose_radius_m"] = float(vehicle_cfg["geometry"]["nose_radius_m"])
    return lim


def point_mass_rhs(vehicle, aero, prop, command_fn, y: np.ndarray, t: float, governor: bool):
    """y = [r(3), v(3), m_ab, m_rocket]. command_fn(t, snap) -> Command."""
    r = y[0:3]
    v = y[3:6]
    m_ab = max(float(y[6]), 0.0)
    m_rk = max(float(y[7]), 0.0)
    m = total_mass(vehicle.cfg, m_ab, m_rk)
    rn = float(np.linalg.norm(r))
    h = rn - A_EARTH
    v_air = air_velocity_eci(v, r)
    Va = float(np.linalg.norm(v_air))
    r_hat = r / rn
    v_hat = v_air / Va if Va > 1.0 else r_hat
    cg = cg_from_nose(vehicle.cfg, m_ab, m_rk)
    # Provisional aero at alpha=0 to let the commander see q, Mach, heating.
    atm = us76(h)
    snap = {
        "t": t,
        "h": h,
        "V_air": Va,
        "r": r,
        "v": v,
        "v_air": v_air,
        "m": m,
        "m_ab": m_ab,
        "m_rocket": m_rk,
        "rho": atm.rho,
        "a": atm.a_mps,
        "mach": Va / atm.a_mps if atm.a_mps > 0 else 0.0,
        "r_hat": r_hat,
        "gamma_air": math.asin(float(np.clip(np.dot(v_hat, r_hat), -1, 1))) if Va > 1 else 0.0,
        "gamma_inertial": math.asin(float(np.clip(np.dot(v / max(np.linalg.norm(v), 1e-9), r_hat), -1, 1))),
        "cg": cg,
    }
    cmd: Command = command_fn(t, snap)
    alpha = 0.0 if cmd.alpha_rad is None else float(cmd.alpha_rad)
    if governor and Va > 1.0:
        # One aero/prop evaluation at the current command, then maybe pull back.
        R = body_triad(v_hat, r_hat, alpha, cmd.bank_rad)
        v_body = R.T @ v_air
        a_act, b_act, _ = alpha_beta_from_body_velocity(v_body)
        ar = aero.compute(h, Va, a_act, b_act, cmd.elevator_rad, cg)
        pr = prop.evaluate(cmd.mode, h, Va, cmd.throttle)
        F = ar.force_body_N + np.array([pr.thrust_N, 0.0, 0.0])
        rep = eval_constraints(_limits_with_rn(vehicle.cfg), rho=atm.rho, V_air=Va, force_body=F, mass=m, alpha_rad=a_act, beta_rad=b_act)
        alpha, th, _notes = apply_governor(alpha, cmd.throttle, rep, vehicle.cfg["limits"])
        cmd = Command(cmd.elevator_rad, th, cmd.mode, alpha, cmd.bank_rad, True)
    if Va > 1.0:
        R = body_triad(v_hat, r_hat, alpha, cmd.bank_rad)
        v_body = R.T @ v_air
        a_act, b_act, _ = alpha_beta_from_body_velocity(v_body)
    else:
        R = np.eye(3)
        a_act, b_act = alpha, 0.0
    ar = aero.compute(h, Va, a_act, b_act, cmd.elevator_rad, cg)
    # Fuel availability
    mode = cmd.mode
    throttle = cmd.throttle
    if mode != "rocket" and m_ab <= 0.0:
        throttle = 0.0
    if mode == "rocket" and m_rk <= 0.0:
        throttle = 0.0
    pr = prop.evaluate(mode, h, Va, throttle)
    cant = math.radians(float(vehicle.cfg["propulsion"].get("thrust_cant_up_deg", 0.0)))
    Ft = pr.thrust_N
    F_thrust = np.array([Ft * math.cos(cant), 0.0, -Ft * math.sin(cant)])
    F_body = ar.force_body_N + F_thrust
    a_eci = grav_acc(r) + (R @ F_body) / m
    mdot_ab = pr.mdot_fuel_kg_s if mode != "rocket" else 0.0
    mdot_rk = pr.mdot_fuel_kg_s if mode == "rocket" else 0.0
    ydot = np.zeros(8)
    ydot[0:3] = v
    ydot[3:6] = a_eci
    ydot[6] = -mdot_ab
    ydot[7] = -mdot_rk
    aux = {
        "h": h,
        "V_air": Va,
        "mach": snap["mach"],
        "alpha": a_act,
        "beta": b_act,
        "gamma_air": snap["gamma_air"],
        "gamma_inertial": snap["gamma_inertial"],
        "mass": m,
        "m_ab": m_ab,
        "m_rocket": m_rk,
        "thrust": Ft,
        "mode": mode,
        "CL": ar.CL,
        "CD": ar.CD,
        "L": ar.L_N,
        "D": ar.D_N,
        "q": ar.dynamic_pressure_Pa,
        "cmd_throttle": throttle,
        "aero_warnings": ar.warnings,
        "prop_warnings": pr.warnings,
        "F_body": F_body,
        "rho": atm.rho,
    }
    return ydot, aux


def rigid_body_rhs(vehicle, aero, prop, command_fn, y: np.ndarray, t: float):
    """y = [r(3), v(3), q(4), omega(3), m_ab, m_rocket] length 15.

    Controls are elevator, throttle, and mode. Alpha is whatever the attitude
    produces (free 6DOF). command_fn may ignore state.
    """
    r = y[0:3]
    v = y[3:6]
    q = y[6:10]
    w = y[10:13]
    m_ab = max(float(y[13]), 0.0)
    m_rk = max(float(y[14]), 0.0)
    m = total_mass(vehicle.cfg, m_ab, m_rk)
    rn = float(np.linalg.norm(r))
    h = rn - A_EARTH
    R = dcm_body_to_inertial(q)
    v_air = air_velocity_eci(v, r)
    v_body = R.T @ v_air
    alpha, beta, Va = alpha_beta_from_body_velocity(v_body)
    cg = cg_from_nose(vehicle.cfg, m_ab, m_rk)
    cmd: Command = command_fn(t, {"h": h, "mach": 0.0, "m_ab": m_ab, "m_rocket": m_rk})
    throttle = cmd.throttle
    mode = cmd.mode
    if mode != "rocket" and m_ab <= 0.0:
        throttle = 0.0
    if mode == "rocket" and m_rk <= 0.0:
        throttle = 0.0
    ar = aero.compute(h, Va, alpha, beta, cmd.elevator_rad, cg)
    pr = prop.evaluate(mode, h, Va, throttle)
    cant = math.radians(float(vehicle.cfg["propulsion"].get("thrust_cant_up_deg", 0.0)))
    Ft = pr.thrust_N
    F_thrust = np.array([Ft * math.cos(cant), 0.0, -Ft * math.sin(cant)])
    # Engine position relative to current CG, body axes. x_from_nose aft distance.
    x_eng = cg - float(vehicle.cfg["propulsion"]["engine_x_from_nose_m"])
    z_eng = -float(vehicle.cfg["propulsion"].get("engine_z_up_m", 0.0))
    r_eng = np.array([x_eng, 0.0, z_eng])
    M_thrust = np.cross(r_eng, F_thrust)
    I = inertia_tensor(vehicle.cfg, m_ab, m_rk)
    mdot_ab = pr.mdot_fuel_kg_s if mode != "rocket" else 0.0
    mdot_rk = pr.mdot_fuel_kg_s if mode == "rocket" else 0.0
    I_dot = inertia_rate(vehicle.cfg, mdot_ab, mdot_rk)
    r_b = R.T @ r
    r_hat_b = r_b / rn
    M_gg = np.zeros(3)
    if vehicle.cfg.get("gravity_gradient", True):
        M_gg = (3.0 * MU_EARTH / rn ** 3) * np.cross(r_hat_b, I @ r_hat_b)
    M = ar.moment_body_Nm + M_thrust + M_gg
    Iw = I @ w
    omega_dot = np.linalg.solve(I, M - np.cross(w, Iw) - I_dot @ w)
    F_body = ar.force_body_N + F_thrust
    ydot = np.zeros(15)
    ydot[0:3] = v
    ydot[3:6] = grav_acc(r) + (R @ F_body) / m
    ydot[6:10] = quat_derivative(q, w)
    ydot[10:13] = omega_dot
    ydot[13] = -mdot_ab
    ydot[14] = -mdot_rk
    aux = {
        "h": h,
        "V_air": Va,
        "V_iner": float(np.linalg.norm(v)),
        "alpha": alpha,
        "beta": beta,
        "mass": m,
        "thrust": Ft,
        "mode": mode,
        "CL": ar.CL,
        "CD": ar.CD,
        "q": ar.dynamic_pressure_Pa,
        "F_body": F_body,
        "M_body": M,
        "rho": us76(h).rho,
        "aero_warnings": ar.warnings,
        "prop_warnings": pr.warnings,
        "mach": Va / us76(h).a_mps if us76(h).a_mps else 0.0,
    }
    return ydot, aux


def integrate(rhs_wrapped, y0, t0, t1, dt, quat_slice=None, stop=None):
    """rhs_wrapped(t, y) -> ydot. Optional stop(t, y, aux) using last aux stored on the wrapper."""
    t = t0
    y = np.array(y0, dtype=float)
    hist = []
    while t < t1 - 1e-9:
        step = min(dt, t1 - t)
        # Grab aux by a side channel
        aux_box = {}

        def rhs(tt, yy, _box=aux_box):
            ydot, aux = rhs_wrapped(tt, yy)
            _box["aux"] = aux
            _box["ydot"] = ydot
            return ydot

        y = rk4_step(rhs, t, y, step, quat_slice=quat_slice)
        # clamp propellant
        if y.shape[0] == 8:
            y[6] = max(y[6], 0.0)
            y[7] = max(y[7], 0.0)
        elif y.shape[0] == 15:
            y[13] = max(y[13], 0.0)
            y[14] = max(y[14], 0.0)
        t = t + step
        aux = dict(aux_box.get("aux", {}))
        aux["t"] = t
        aux["y"] = y.copy()
        hist.append(aux)
        if stop is not None and stop(t, y, aux):
            break
    return hist
