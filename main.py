"""Command line and interactive menu.

    python main.py --self-test
    python main.py --vehicle data/example_vehicle.json
    python main.py --vehicle data/example_vehicle.json --task mission
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spaceplane.aerodynamics import Aerodynamics  # noqa: E402
from spaceplane.atmosphere import us76  # noqa: E402
from spaceplane.guidance import select_mode  # noqa: E402
from spaceplane.mission import optimize_steering, run_6dof, run_mission  # noqa: E402
from spaceplane.performance import breguet_range, envelope, ground_roll, point, trim_level  # noqa: E402
from spaceplane.reporting import format_summary, write_envelope_csv, write_history_csv  # noqa: E402
from spaceplane.types import Command  # noqa: E402
from spaceplane.vehicle import Vehicle  # noqa: E402


def cmd_self_test() -> int:
    from tests.test_physics import main as test_main

    return test_main()


def cmd_atmosphere(h: float) -> None:
    a = us76(h)
    print(f"h_m {a.h_m:.1f}")
    print(f"geopotential_m {a.geopotential_m:.1f}")
    print(f"T_K {a.T_K:.3f}")
    print(f"p_Pa {a.p_Pa:.3f}")
    print(f"rho {a.rho:.6f}")
    print(f"a_mps {a.a_mps:.3f}")
    print(f"mu {a.mu:.6e}")
    print(f"in_US76_lapse_model {a.in_model}")


def cmd_aero(v: Vehicle, h: float, mach: float, alpha_deg: float) -> None:
    atm = us76(h)
    V = mach * atm.a_mps
    m = v.cfg["mass"]
    from spaceplane.massprops import cg_from_nose

    cg = cg_from_nose(v.cfg, m["fuel_airbreathing_kg"], m["rocket_propellant_kg"])
    ar = v.aero.compute(h, V, math.radians(alpha_deg), 0.0, 0.0, cg)
    print(f"Mach {ar.mach:.3f}  q_Pa {ar.dynamic_pressure_Pa:.1f}  Re {ar.reynolds:.3e}  Cf {ar.Cf:.5f}")
    print(f"CL {ar.CL:.4f}  CD {ar.CD:.4f}  Cm_noseup {ar.Cm_noseup:.4f}")
    print(f"L_N {ar.L_N:.1f}  D_N {ar.D_N:.1f}")
    print(f"force_body {ar.force_body_N}")
    print(f"moment_body {ar.moment_body_Nm}")
    for n in ar.warnings:
        print("note:", n)


def cmd_prop(v: Vehicle, h: float, mach: float, mode: str, throttle: float) -> None:
    atm = us76(h)
    pr = v.prop.evaluate(mode, h, mach * atm.a_mps, throttle)
    print(f"mode {pr.mode} thrust_N {pr.thrust_N:.1f} mdot_fuel {pr.mdot_fuel_kg_s:.3f} mdot_air {pr.mdot_air_kg_s:.3f}")
    print(f"Isp_s {pr.isp_s:.2f} TSFC_kg_per_Ns {pr.tsfc_kg_Ns} Me {pr.mach_exit}")
    print(f"stations {pr.stations}")
    for n in pr.warnings:
        print("note:", n)


def cmd_trim(v: Vehicle, h: float, mach: float, mode: str) -> None:
    atm = us76(h)
    m = v.cfg["mass"]
    tr = trim_level(v, h, mach * atm.a_mps, mode, m["fuel_airbreathing_kg"], m["rocket_propellant_kg"])
    print(f"alpha_deg {math.degrees(tr['alpha_rad']):.2f}")
    print(f"elevator_deg {math.degrees(tr['elevator_rad']):.2f}")
    print(f"throttle {tr['throttle']:.3f}")
    print(f"vertical_residual_N {tr['vertical_residual_N']:.1f}")
    print(f"axial_residual_N {tr['axial_residual_N']:.1f}")
    print(f"My_Nm {tr['My_Nm']:.1f}")
    print("Flat-Earth trim (V^2/r neglected). Residuals near zero means a trimmed point exists.")


def cmd_performance(v: Vehicle) -> None:
    m = v.cfg["mass"]
    prop = v.cfg["propulsion"]

    def mode_fn(mach, h):
        atm = us76(float(h))
        return select_mode(mach, atm.rho, m["fuel_airbreathing_kg"], m["rocket_propellant_kg"], prop)

    machs = [0.5, 0.8, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    alts = [0.0, 5000.0, 10000.0, 15000.0, 20000.0, 30000.0]
    rows = envelope(v, m["fuel_airbreathing_kg"], m["rocket_propellant_kg"], machs, alts, mode_fn)
    path = ROOT / "outputs" / "envelope.csv"
    write_envelope_csv(path, rows)
    print(f"wrote {path}")
    # One Breguet estimate at the best Ps point near 10 km, Mach 0.8, if feasible.
    atm = us76(10000.0)
    row = point(
        v,
        10000.0,
        0.8 * atm.a_mps,
        math.radians(2.0),
        0.0,
        0.0,
        1.0,
        "turbojet",
        m["fuel_airbreathing_kg"],
        m["rocket_propellant_kg"],
    )
    R = breguet_range(
        0.8 * atm.a_mps,
        row["L_over_D"] if row["L_over_D"] == row["L_over_D"] else -1,
        row["prop"].thrust_N,
        row["prop"].mdot_fuel_kg_s,
        row["mass"],
        row["mass"] - 0.5 * m["fuel_airbreathing_kg"],
    )
    print(f"sample Ps at 10 km Mach 0.8 alpha 2 deg: {row['Ps_m_s']:.2f} m/s")
    print(f"sample L/D {row['L_over_D']}")
    print(f"Breguet range for burning half the airbreathing fuel at that frozen point: {R}")


def cmd_ground(v: Vehicle) -> None:
    m = v.cfg["mass"]
    cl_max = float(v.cfg["geometry"].get("cl_max", 1.2))
    out = ground_roll(v, 0.0, "turbojet", m["fuel_airbreathing_kg"], m["rocket_propellant_kg"], mu_r=0.03, cl_max=cl_max)
    print(out)
    print("mu_r = 0.03 is a user rolling-friction input, not a measured value.")
    print("CL_max is a user input. The section model does not predict stall.")


def cmd_mission(v: Vehicle, optimize: bool) -> None:
    if optimize:
        hist, summary = optimize_steering(v)
        name = "optimize.csv"
    else:
        hist, summary = run_mission(v)
        name = "mission.csv"
    path = ROOT / "outputs" / name
    write_history_csv(path, hist)
    print(format_summary(summary))
    print(f"wrote {path} ({len(hist)} rows)")
    print("If stop_reason is not orbit_target_met, the example vehicle did not reach the target. That is a feasible outcome of these inputs.")


def cmd_6dof(v: Vehicle) -> None:
    mode = v.cfg.get("sim6dof", {}).get("mode", "turbojet")
    dt = float(v.cfg.get("sim6dof", {}).get("dt_s", 0.05))
    t_max = float(v.cfg.get("sim6dof", {}).get("t_max_s", 20.0))
    # throttle < 0 asks run_6dof to trim first
    hist, trim, cmd = run_6dof(v, Command(0.0, -1.0, mode, None, 0.0, False), t_max, dt)
    path = ROOT / "outputs" / "sim6dof.csv"
    write_history_csv(path, hist)
    print(f"mode {cmd.mode} throttle {cmd.throttle:.3f} elevator_deg {math.degrees(cmd.elevator_rad):.2f}")
    if trim is not None:
        print(f"trim alpha_deg {math.degrees(trim['alpha_rad']):.2f}")
        print(f"vertical residual N {trim['vertical_residual_N']:.1f}  axial residual N {trim['axial_residual_N']:.1f}")
    if hist:
        print(f"t {hist[-1]['t']:.2f} alt {hist[-1]['h']:.1f} alpha_deg {math.degrees(hist[-1]['alpha']):.2f}")
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Supersonic / spaceplane 6DOF preliminary design tool")
    p.add_argument("--vehicle", default=str(ROOT / "data" / "example_vehicle.json"))
    p.add_argument(
        "--task",
        default="menu",
        choices=["menu", "self-test", "atmosphere", "aero", "propulsion", "trim", "performance", "groundroll", "mission", "optimize", "sim6dof"],
    )
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--altitude", type=float, default=15000.0)
    p.add_argument("--mach", type=float, default=2.0)
    p.add_argument("--alpha-deg", type=float, default=4.0)
    p.add_argument("--mode", default="ramjet")
    p.add_argument("--throttle", type=float, default=1.0)
    return p


def menu(v: Vehicle) -> None:
    actions = {
        "1": "self-test",
        "2": "atmosphere",
        "3": "aero",
        "4": "propulsion",
        "5": "trim",
        "6": "performance",
        "7": "groundroll",
        "8": "sim6dof",
        "9": "mission",
        "10": "optimize",
        "0": "quit",
    }
    print("Vehicle:", v.cfg.get("name"), "from", v.source)
    print(v.cfg.get("disclaimer", ""))
    while True:
        print(
            """
1 self-test validation
2 atmosphere at an altitude
3 aero polar point
4 propulsion point
5 level-flight trim
6 performance envelope + Breguet sample
7 ground roll
8 free 6DOF (trimmed controls, then hands off)
9 mission: energy climb + gravity turn
10 optimize gravity-turn parameters
0 quit
"""
        )
        choice = input("select> ").strip()
        task = actions.get(choice, choice)
        if task in ("0", "quit"):
            return
        if task == "self-test":
            cmd_self_test()
            continue
        if task == "atmosphere":
            h = float(input("altitude m [11000]: ") or "11000")
            cmd_atmosphere(h)
            continue
        if task == "aero":
            h = float(input("altitude m [15000]: ") or "15000")
            M = float(input("Mach [2]: ") or "2")
            a = float(input("alpha deg [4]: ") or "4")
            cmd_aero(v, h, M, a)
            continue
        if task == "propulsion":
            h = float(input("altitude m [15000]: ") or "15000")
            M = float(input("Mach [2]: ") or "2")
            mode = input("mode turbojet|ramjet|scramjet|rocket [ramjet]: ") or "ramjet"
            th = float(input("throttle [1]: ") or "1")
            cmd_prop(v, h, M, mode, th)
            continue
        if task == "trim":
            h = float(input("altitude m [15000]: ") or "15000")
            M = float(input("Mach [1.7]: ") or "1.7")
            mode = input("mode [turbojet]: ") or "turbojet"
            cmd_trim(v, h, M, mode)
            continue
        dispatch_task(task, v, argparse.Namespace(altitude=15000, mach=2, alpha_deg=4, mode="ramjet", throttle=1))


def dispatch_task(task: str, v: Vehicle | None, ns) -> int:
    if task == "self-test":
        return cmd_self_test()
    if v is None:
        raise SystemExit("a vehicle JSON is required")
    if task == "atmosphere":
        cmd_atmosphere(ns.altitude)
    elif task == "aero":
        cmd_aero(v, ns.altitude, ns.mach, ns.alpha_deg)
    elif task == "propulsion":
        cmd_prop(v, ns.altitude, ns.mach, ns.mode, ns.throttle)
    elif task == "trim":
        cmd_trim(v, ns.altitude, ns.mach, ns.mode)
    elif task == "performance":
        cmd_performance(v)
    elif task == "groundroll":
        cmd_ground(v)
    elif task == "mission":
        cmd_mission(v, optimize=False)
    elif task == "optimize":
        cmd_mission(v, optimize=True)
    elif task == "sim6dof":
        cmd_6dof(v)
    elif task == "menu":
        menu(v)
    else:
        raise SystemExit(f"unknown task {task}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.self_test or ns.task == "self-test":
        return cmd_self_test()
    v = Vehicle.from_json(ns.vehicle)
    return dispatch_task(ns.task, v, ns)


if __name__ == "__main__":
    raise SystemExit(main())
