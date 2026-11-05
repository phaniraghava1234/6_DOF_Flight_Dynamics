from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .constants import A_EARTH
from .frames import ecef_spherical_lat_lon, eci_to_ecef


def write_history_csv(path: Path, hist: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t_s",
        "x_m",
        "y_m",
        "z_m",
        "lat_deg",
        "lon_deg",
        "alt_m",
        "V_air_m_s",
        "mach",
        "alpha_deg",
        "gamma_air_deg",
        "mass_kg",
        "fuel_ab_kg",
        "rocket_kg",
        "thrust_N",
        "mode",
        "CL",
        "CD",
        "q_Pa",
        "qdot_W_m2",
        "n_total",
        "T_rad_K",
        "violations",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in hist:
            y = row["y"]
            r = y[0:3]
            rec = eci_to_ecef(r, row["t"])
            lat, lon = ecef_spherical_lat_lon(rec)
            # 3DOF state stores masses at 6,7. 6DOF stores them at 13,14.
            if len(y) == 8:
                m_ab, m_rk = y[6], y[7]
                gamma = row.get("gamma_air", float("nan"))
            else:
                m_ab, m_rk = y[13], y[14]
                v = y[3:6]
                gamma = math.asin(float(np.clip(np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r) + 1e-9), -1, 1)))
            w.writerow(
                {
                    "t_s": row["t"],
                    "x_m": r[0],
                    "y_m": r[1],
                    "z_m": r[2],
                    "lat_deg": math.degrees(lat),
                    "lon_deg": math.degrees(lon),
                    "alt_m": row["h"],
                    "V_air_m_s": row.get("V_air", ""),
                    "mach": row.get("mach", ""),
                    "alpha_deg": math.degrees(row.get("alpha", 0.0)),
                    "gamma_air_deg": math.degrees(gamma) if gamma == gamma else "",
                    "mass_kg": row.get("mass", ""),
                    "fuel_ab_kg": m_ab,
                    "rocket_kg": m_rk,
                    "thrust_N": row.get("thrust", ""),
                    "mode": row.get("mode", ""),
                    "CL": row.get("CL", ""),
                    "CD": row.get("CD", ""),
                    "q_Pa": row.get("q", ""),
                    "qdot_W_m2": row.get("qdot", ""),
                    "n_total": row.get("n_total", ""),
                    "T_rad_K": row.get("T_rad", ""),
                    "violations": "|".join(row.get("violations", [])),
                }
            )


def write_envelope_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def format_summary(summary: dict) -> str:
    el = summary.get("elements", {})
    lines = [
        f"t_final_s: {summary.get('t_final_s')}",
        f"alt_final_m: {summary.get('alt_final_m')}",
        f"mass_final_kg: {summary.get('mass_final_kg')}",
        f"propellant_used_kg: {summary.get('propellant_used_kg')}",
        f"fuel_ab_remaining_kg: {summary.get('fuel_ab_remaining_kg')}",
        f"rocket_remaining_kg: {summary.get('rocket_remaining_kg')}",
        f"q_peak_Pa: {summary.get('q_peak_Pa')}",
        f"qdot_peak_W_m2: {summary.get('qdot_peak_W_m2')}",
        f"n_peak: {summary.get('n_peak')}",
        f"violations_seen: {summary.get('any_violation')}",
        f"stop_reason: {summary.get('stop_reason')}",
    ]
    if el:
        lines += [
            f"specific_energy: {el.get('specific_energy_m2_s2')}",
            f"sma_m: {el.get('sma_m')}",
            f"eccentricity: {el.get('eccentricity')}",
            f"inclination_deg: {math.degrees(el.get('inclination_rad', 0.0))}",
            f"rp_m: {el.get('rp_m')}",
            f"ra_m: {el.get('ra_m')}",
            f"rp_alt_m: {None if el.get('rp_m') != el.get('rp_m') else el.get('rp_m') - A_EARTH}",
            f"ra_alt_m: {None if el.get('ra_m') == float('inf') else (None if el.get('ra_m') != el.get('ra_m') else el.get('ra_m') - A_EARTH)}",
        ]
    if "optimizer" in summary:
        lines.append(f"optimizer: {summary['optimizer']}")
    if "guidance" in summary:
        lines.append(f"guidance: {summary['guidance']}")
    return "\n".join(lines)
