"""Mission integration and steering-parameter optimization.

The integrated 3DOF trajectory is the mission analysis. Orbital elements at
the end are two-body osculating elements (Vallado). Peaks of q, heating, and
load factor are the constraint report.

optimize_steering uses SLSQP (SciPy) on three gravity-turn parameters.
That is parameterized direct shooting (Betts), not a full optimal-control
transcription. If SciPy is absent, the un-optimized guidance is flown and
the gap is reported.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import A_EARTH, MU_EARTH
from .dynamics import integrate, point_mass_rhs, rigid_body_rhs
from .frames import (
    air_velocity_eci,
    body_triad,
    dcm_to_quaternion,
    ned_to_ecef_dcm,
    spherical_position,
)
from .guidance import AscentGuidance, GravityTurn
from .gravity import circular_speed, orbital_elements
from .massprops import total_mass
from .types import Command


def initial_point_mass_state(cfg: dict) -> np.ndarray:
    ic = cfg["initial_condition"]
    lat = math.radians(float(ic["latitude_deg"]))
    lon = math.radians(float(ic["longitude_deg"]))
    h = float(ic["altitude_m"])
    V = float(ic["airspeed_m_s"])
    gamma = math.radians(float(ic["flight_path_deg"]))
    chi = math.radians(float(ic["heading_deg"]))
    r = spherical_position(lat, lon, h)
    v_ned = V * np.array(
        [math.cos(gamma) * math.cos(chi), math.cos(gamma) * math.sin(chi), -math.sin(gamma)],
        dtype=float,
    )
    v_ecef = ned_to_ecef_dcm(lat, lon) @ v_ned
    # t = 0: ECI = ECEF, and airspeed is Earth-relative, so add Earth rotation.
    v_eci = v_ecef + np.cross(np.array([0.0, 0.0, 7.292115e-5]), r)
    m = cfg["mass"]
    return np.array(
        [r[0], r[1], r[2], v_eci[0], v_eci[1], v_eci[2], float(m["fuel_airbreathing_kg"]), float(m["rocket_propellant_kg"])],
        dtype=float,
    )


def _enrich(snap_from_aux, y):
    return y


def run_point_mass(vehicle, command_fn, t_max, dt, governor=True, stop_on_orbit=True):
    y = initial_point_mass_state(vehicle.cfg)
    h_target = float(vehicle.cfg["mission"]["h_orbit_m"])

    def wrapped(t, yy):
        return point_mass_rhs(vehicle, vehicle.aero, vehicle.prop, command_fn, yy, t, governor)

    def stop(t, yy, aux):
        if aux["h"] <= 0.0 and t > 1.0:
            aux["stop_reason"] = "impact"
            return True
        r = yy[0:3]
        v = yy[3:6]
        rn = float(np.linalg.norm(r))
        V = float(np.linalg.norm(v))
        gamma_i = math.asin(float(np.clip(np.dot(v, r) / (V * rn), -1, 1)))
        if stop_on_orbit and aux["h"] >= h_target and abs(gamma_i) < math.radians(2.0) and V >= 0.98 * circular_speed(rn):
            aux["stop_reason"] = "orbit_target_met"
            return True
        if yy[6] <= 0.0 and yy[7] <= 0.0 and aux["h"] < h_target:
            # Keep coasting until t_max; do not stop only because tanks are dry.
            return False
        return False

    hist = integrate(wrapped, y, 0.0, t_max, dt, quat_slice=None, stop=stop)
    return hist


def summarize(vehicle, hist: list[dict]) -> dict:
    if not hist:
        return {"error": "empty trajectory"}
    last = hist[-1]
    y = last["y"]
    r, v = y[0:3], y[3:6]
    el = orbital_elements(r, v, MU_EARTH)
    q_peak = max(h["q"] for h in hist)
    # Heating and load factor need a second pass stored if present
    def qdot_of(hrow):
        return hrow.get("qdot", 0.0)

    # Recompute peaks from stored force if the constraint object was not stored.
    from .constraints import evaluate as ev
    from .atmosphere import us76

    qdots = []
    ns = []
    for hrow in hist:
        atm = us76(hrow["h"])
        rep = ev(
            {**vehicle.cfg["limits"], "nose_radius_m": vehicle.cfg["geometry"]["nose_radius_m"]},
            rho=atm.rho,
            V_air=hrow["V_air"],
            force_body=hrow["F_body"],
            mass=hrow["mass"],
            alpha_rad=hrow["alpha"],
            beta_rad=hrow.get("beta", 0.0),
        )
        qdots.append(rep.qdot_W_m2)
        ns.append(rep.n_total)
        hrow["qdot"] = rep.qdot_W_m2
        hrow["n_total"] = rep.n_total
        hrow["violations"] = list(rep.violated)
        hrow["T_rad"] = rep.t_rad_K
    m0 = total_mass(vehicle.cfg, vehicle.cfg["mass"]["fuel_airbreathing_kg"], vehicle.cfg["mass"]["rocket_propellant_kg"])
    return {
        "t_final_s": last["t"],
        "alt_final_m": last["h"],
        "V_air_final": last["V_air"],
        "mach_final": last["mach"],
        "mass_final_kg": last["mass"],
        "propellant_used_kg": m0 - last["mass"],
        "fuel_ab_remaining_kg": last["m_ab"],
        "rocket_remaining_kg": last["m_rocket"],
        "q_peak_Pa": q_peak,
        "qdot_peak_W_m2": max(qdots) if qdots else 0.0,
        "n_peak": max(ns) if ns else 0.0,
        "stop_reason": last.get("stop_reason", "t_max"),
        "elements": el,
        "any_violation": sorted({v for hrow in hist for v in hrow["violations"]}),
    }


def run_mission(vehicle, alpha_pull_deg=None, gamma_switch_deg=None, alpha_circ_deg=None, governor=True):
    mcfg = vehicle.cfg["mission"]
    ap = math.radians(float(mcfg["pullup_alpha_deg"] if alpha_pull_deg is None else alpha_pull_deg))
    gs = math.radians(float(mcfg["pullup_gamma_deg"] if gamma_switch_deg is None else gamma_switch_deg))
    ac = math.radians(float(mcfg["circularize_alpha_deg"] if alpha_circ_deg is None else alpha_circ_deg))
    rocket = GravityTurn(ap, gs, ac, float(mcfg["h_orbit_m"]))
    guide = AscentGuidance(vehicle, rocket)
    hist = run_point_mass(vehicle, guide, float(mcfg["t_max_s"]), float(mcfg["dt_s"]), governor=governor)
    summary = summarize(vehicle, hist)
    summary["guidance"] = {
        "alpha_pull_deg": math.degrees(ap),
        "gamma_switch_deg": math.degrees(gs),
        "alpha_circ_deg": math.degrees(ac),
    }
    return hist, summary


def optimize_steering(vehicle):
    try:
        from scipy.optimize import minimize
    except ImportError:
        hist, summary = run_mission(vehicle)
        summary["optimizer"] = "scipy missing; flew the JSON guidance without tuning"
        return hist, summary

    lim = vehicle.cfg["limits"]
    h_target = float(vehicle.cfg["mission"]["h_orbit_m"])
    evals = {"n": 0}

    def objective(x):
        evals["n"] += 1
        hist, summary = run_mission(vehicle, x[0], x[1], x[2], governor=True)
        el = summary["elements"]
        r_last = hist[-1]["y"][0:3]
        v_last = hist[-1]["y"][3:6]
        rn = float(np.linalg.norm(r_last))
        V = float(np.linalg.norm(v_last))
        gamma = math.asin(float(np.clip(np.dot(v_last, r_last) / (V * rn), -1, 1)))
        v_err = (V - circular_speed(rn)) / circular_speed(rn)
        h_err = (summary["alt_final_m"] - h_target) / h_target
        # Maximize remaining mass, penalize missing the orbit and constraint peaks.
        pen = 0.0
        pen += 50.0 * max(-h_err, 0.0) ** 2
        pen += 20.0 * v_err ** 2
        pen += 10.0 * (gamma / 0.1) ** 2
        pen += 5.0 * max(summary["q_peak_Pa"] / lim["q_max_pa"] - 1.0, 0.0) ** 2
        pen += 5.0 * max(summary["qdot_peak_W_m2"] / lim["qdot_max_w_m2"] - 1.0, 0.0) ** 2
        pen += 5.0 * max(summary["n_peak"] / lim["n_max"] - 1.0, 0.0) ** 2
        # Remaining useful mass (inert + residuals). Higher is better.
        return -(summary["mass_final_kg"]) / 1000.0 + 1000.0 * pen

    x0 = np.array(
        [
            float(vehicle.cfg["mission"]["pullup_alpha_deg"]),
            float(vehicle.cfg["mission"]["pullup_gamma_deg"]),
            float(vehicle.cfg["mission"]["circularize_alpha_deg"]),
        ],
        dtype=float,
    )
    bounds = [
        (0.0, float(lim["alpha_max_deg"])),
        (5.0, 40.0),
        (-5.0, float(lim["alpha_max_deg"])),
    ]
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, options={"maxiter": 12, "ftol": 1e-4, "disp": False})
    hist, summary = run_mission(vehicle, res.x[0], res.x[1], res.x[2], governor=True)
    summary["optimizer"] = {
        "success": bool(res.success),
        "message": str(res.message),
        "evaluations": evals["n"],
        "x_deg": [float(res.x[0]), float(res.x[1]), float(res.x[2])],
    }
    return hist, summary


def initial_6dof_state(cfg: dict, alpha: float, elevator: float) -> np.ndarray:
    y3 = initial_point_mass_state(cfg)
    r, v = y3[0:3], y3[3:6]
    v_air = air_velocity_eci(v, r)
    Va = float(np.linalg.norm(v_air))
    v_hat = v_air / Va
    r_hat = r / np.linalg.norm(r)
    R = body_triad(v_hat, r_hat, alpha, 0.0)
    q = dcm_to_quaternion(R)
    y = np.zeros(15)
    y[0:6] = y3[0:6]
    y[6:10] = q
    y[10:13] = 0.0
    y[13] = y3[6]
    y[14] = y3[7]
    return y


def run_6dof(vehicle, command: Command, t_max: float, dt: float):
    from .performance import trim_level

    # If the caller passes throttle < 0, interpret that as "please trim".
    y = None
    if command.throttle < 0.0:
        ic = vehicle.cfg["initial_condition"]
        trim = trim_level(
            vehicle,
            float(ic["altitude_m"]),
            float(ic["airspeed_m_s"]),
            command.mode,
            float(vehicle.cfg["mass"]["fuel_airbreathing_kg"]),
            float(vehicle.cfg["mass"]["rocket_propellant_kg"]),
        )
        command = Command(
            elevator_rad=trim["elevator_rad"],
            throttle=max(trim["throttle"], 0.0),
            mode=command.mode,
            alpha_rad=None,
            kinematic_alpha=False,
        )
        y = initial_6dof_state(vehicle.cfg, trim["alpha_rad"], trim["elevator_rad"])
        trim_info = trim
    else:
        alpha0 = 0.0 if command.alpha_rad is None else command.alpha_rad
        y = initial_6dof_state(vehicle.cfg, alpha0, command.elevator_rad)
        trim_info = None

    def command_fn(t, snap):
        return command

    def wrapped(t, yy):
        return rigid_body_rhs(vehicle, vehicle.aero, vehicle.prop, command_fn, yy, t)

    def stop(t, yy, aux):
        if aux["h"] <= 0.0 and t > 0.5:
            aux["stop_reason"] = "impact"
            return True
        return False

    hist = integrate(wrapped, y, 0.0, t_max, dt, quat_slice=slice(6, 10), stop=stop)
    return hist, trim_info, command
