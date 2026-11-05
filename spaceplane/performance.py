"""Point performance.

Ps = (T cos(alpha) - D) V / W    (Rutowski), reported in m/s.
Also reported: exact specific excess power (F_aero+thrust · v_air) / (m g0).

Breguet (Anderson, Aircraft Performance and Design), frozen condition:
    c = (mdot_fuel * g0) / F     weight of fuel per unit time, per unit thrust
    R = (V / c) (L/D) ln(m0/m1)

Ground roll (Anderson): integrate
    m dV/dt = T - D - mu_r (W - L)
from rest to 1.2 Vstall, with Vstall = sqrt(2 W / (rho S CL_max)).
CL_max and mu_r are user inputs. The section model does not predict stall.
"""

from __future__ import annotations

import math

import numpy as np

from .atmosphere import us76
from .constants import A_EARTH, G0
from .constraints import evaluate as eval_constraints
from .massprops import cg_from_nose, total_mass


def _limits(cfg):
    lim = dict(cfg["limits"])
    lim["nose_radius_m"] = float(cfg["geometry"]["nose_radius_m"])
    return lim


def point(vehicle, h, V, alpha, beta, elevator, throttle, mode, m_ab, m_rocket):
    cg = cg_from_nose(vehicle.cfg, m_ab, m_rocket)
    aero = vehicle.aero.compute(h, V, alpha, beta, elevator, cg)
    prop = vehicle.prop.evaluate(mode, h, V, throttle)
    m = total_mass(vehicle.cfg, m_ab, m_rocket)
    W = m * G0
    cant = math.radians(float(vehicle.cfg["propulsion"].get("thrust_cant_up_deg", 0.0)))
    T = prop.thrust_N
    D = aero.D_N
    L = aero.L_N
    Ps = ((T * math.cos(alpha + cant) - D) * V / W) if W > 0 and V > 0 else 0.0
    atm = us76(h)
    F_body = aero.force_body_N + np.array([T * math.cos(cant), 0.0, -T * math.sin(cant)])
    rep = eval_constraints(_limits(vehicle.cfg), rho=atm.rho, V_air=V, force_body=F_body, mass=m, alpha_rad=alpha, beta_rad=beta)
    ld = L / D if abs(D) > 1.0 else float("nan")
    return {
        "aero": aero,
        "prop": prop,
        "Ps_m_s": Ps,
        "L_over_D": ld,
        "mass": m,
        "weight": W,
        "constraints": rep,
        "mach": V / atm.a_mps if atm.a_mps else 0.0,
        "q": aero.dynamic_pressure_Pa,
    }


def ps_sweep(vehicle, h, V, mode, throttle, m_ab, m_rocket, n=9):
    a_max = math.radians(float(vehicle.cfg["limits"]["alpha_max_deg"]))
    best = None
    rows = []
    for alpha in np.linspace(-a_max, a_max, n):
        row = point(vehicle, h, V, float(alpha), 0.0, 0.0, throttle, mode, m_ab, m_rocket)
        row["alpha"] = float(alpha)
        feasible = len(row["constraints"].violated) == 0 and row["prop"].thrust_N > 0.0
        row["feasible"] = feasible
        rows.append(row)
        if feasible and (best is None or row["Ps_m_s"] > best["Ps_m_s"]):
            best = row
    if best is None and rows:
        # Least violation, then highest Ps
        def key(r):
            return (len(r["constraints"].violated), -r["Ps_m_s"])

        best = sorted(rows, key=key)[0]
    return best, rows


def breguet_range(V, L_over_D, thrust, mdot_fuel, m0, m1) -> float:
    if thrust <= 0 or mdot_fuel <= 0 or L_over_D <= 0 or m1 <= 0 or m0 <= m1:
        return float("nan")
    c = (mdot_fuel * G0) / thrust
    return (V / c) * L_over_D * math.log(m0 / m1)


def envelope(vehicle, m_ab, m_rocket, machs, alts, mode_fn):
    rows = []
    for h in alts:
        atm = us76(float(h))
        for M in machs:
            V = M * atm.a_mps
            mode = mode_fn(M, h)
            best, _ = ps_sweep(vehicle, float(h), V, mode, 1.0, m_ab, m_rocket, n=7)
            rows.append(
                {
                    "alt_m": float(h),
                    "mach": float(M),
                    "mode": mode,
                    "Ps_m_s": None if best is None else best["Ps_m_s"],
                    "alpha_deg": None if best is None else math.degrees(best["alpha"]),
                    "CL": None if best is None else best["aero"].CL,
                    "CD": None if best is None else best["aero"].CD,
                    "thrust_N": None if best is None else best["prop"].thrust_N,
                    "feasible": None if best is None else best["feasible"],
                    "q_Pa": None if best is None else best["q"],
                    "qdot": None if best is None else best["constraints"].qdot_W_m2,
                    "violations": "" if best is None else "|".join(best["constraints"].violated),
                }
            )
    return rows


def trim_level(vehicle, h, V, mode, m_ab, m_rocket):
    """Flat-Earth trim (Etkin / Anderson): Cm = 0, T cos(alpha) = D, L + T sin(alpha) = W.

    Valid when V^2/r is negligible next to g. Residuals are returned; a failed
    trim is a result, not an exception.
    """
    a_max = math.radians(float(vehicle.cfg["limits"]["alpha_max_deg"]))
    e_max = math.radians(float(vehicle.cfg["limits"]["elevator_max_deg"]))
    m = total_mass(vehicle.cfg, m_ab, m_rocket)
    W = m * G0 * (A_EARTH / (A_EARTH + h)) ** 2
    best = None
    for alpha in np.linspace(-a_max, a_max, 13):
        elev = _solve_elevator(vehicle, h, V, float(alpha), e_max, m_ab, m_rocket)
        row = point(vehicle, h, V, float(alpha), 0.0, elev, 1.0, mode, m_ab, m_rocket)
        Tavail = max(row["prop"].thrust_N, 0.0)
        D = row["aero"].D_N
        need = D / max(math.cos(float(alpha)), 0.2)
        throttle = float(np.clip(need / Tavail, 0.0, 1.0)) if Tavail > 1.0 else 1.0
        row2 = point(vehicle, h, V, float(alpha), 0.0, elev, throttle, mode, m_ab, m_rocket)
        T = row2["prop"].thrust_N
        L = row2["aero"].L_N
        vertical = L + T * math.sin(float(alpha)) - W
        axial = T * math.cos(float(alpha)) - row2["aero"].D_N
        moment = row2["aero"].moment_body_Nm[1]
        score = abs(vertical) / W + abs(axial) / W + abs(moment) / (W * float(vehicle.cfg["geometry"]["wing_chord_m"]))
        cand = {
            "alpha_rad": float(alpha),
            "elevator_rad": elev,
            "throttle": throttle,
            "mode": mode,
            "vertical_residual_N": vertical,
            "axial_residual_N": axial,
            "My_Nm": moment,
            "score": score,
            "point": row2,
        }
        if best is None or score < best["score"]:
            best = cand
    return best


def _solve_elevator(vehicle, h, V, alpha, e_max, m_ab, m_rocket):
    def my(e):
        cg = cg_from_nose(vehicle.cfg, m_ab, m_rocket)
        ar = vehicle.aero.compute(h, V, alpha, 0.0, e, cg)
        return ar.moment_body_Nm[1]

    lo, hi = -e_max, e_max
    flo, fhi = my(lo), my(hi)
    if flo * fhi > 0.0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        fm = my(mid)
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def ground_roll(vehicle, h, mode, m_ab, m_rocket, mu_r, cl_max):
    m = total_mass(vehicle.cfg, m_ab, m_rocket)
    W = m * G0
    atm = us76(h)
    Vstall = math.sqrt(2.0 * W / (atm.rho * float(vehicle.cfg["geometry"]["wing_area_m2"]) * cl_max))
    Vrot = 1.2 * Vstall
    V = 0.1
    s = 0.0
    t = 0.0
    dt = 0.2
    while V < Vrot and t < 120.0:
        # Ground: alpha ~ 0, lift from the section at alpha 0 (usually ~0 unless flap).
        row = point(vehicle, h, V, 0.0, 0.0, 0.0, 1.0, mode, m_ab, m_rocket)
        L = max(row["aero"].L_N, 0.0)
        D = max(row["aero"].D_N, 0.0)
        T = max(row["prop"].thrust_N, 0.0)
        a = (T - D - mu_r * max(W - L, 0.0)) / m
        if a <= 0.05 and V < 2.0:
            return {"distance_m": float("inf"), "Vstall": Vstall, "reason": "thrust does not exceed static resistance"}
        V = V + a * dt
        s = s + V * dt
        t = t + dt
    return {"distance_m": s, "time_s": t, "Vstall_m_s": Vstall, "Vrot_m_s": Vrot, "mu_r": mu_r, "CL_max": cl_max}
