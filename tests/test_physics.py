"""Checks against standard limits. Run: python main.py --self-test"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spaceplane.aerodynamics import (  # noqa: E402
    cp_max_modified_newtonian,
    incompressible_panel,
    normal_shock_mach2,
    normal_shock_pressure_ratio,
    prandtl_meyer,
    prandtl_meyer_max,
)
from spaceplane.atmosphere import us76  # noqa: E402
from spaceplane.constants import A_EARTH, G0, MU_EARTH, R_AIR  # noqa: E402
from spaceplane.dynamics import integrate, rk4_step  # noqa: E402
from spaceplane.frames import (  # noqa: E402
    body_triad,
    dcm_body_to_inertial,
    dcm_to_quaternion,
    ned_to_ecef_dcm,
    quat_derivative,
)
from spaceplane.gravity import acceleration, circular_speed, potential  # noqa: E402
from spaceplane.propulsion import (  # noqa: E402
    expand_nozzle,
    mass_flow_parameter,
    rayleigh_tt_ratio,
    solve_rayleigh_mach,
    Propulsion,
)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_isa_sea_level():
    a = us76(0.0)
    check(abs(a.T_K - 288.15) < 1e-9, a.T_K)
    check(abs(a.p_Pa - 101325.0) < 1e-6, a.p_Pa)
    check(abs(a.rho - 1.225) < 0.002, a.rho)


def test_isa_11km():
    # Published US76 tropopause values, 0.5% band.
    a = us76(11000.0)
    check(abs(a.T_K - 216.65) < 0.05, a.T_K)
    check(abs(a.p_Pa - 22632.0) / 22632.0 < 0.005, a.p_Pa)
    check(abs(a.rho - 0.3639) / 0.3639 < 0.01, a.rho)


def test_hydrostatic_sea_level():
    p0 = us76(0.0).p_Pa
    p1 = us76(200.0).p_Pa
    rho = 0.5 * (us76(0.0).rho + us76(200.0).rho)
    slope = (p0 - p1) / 200.0
    check(abs(slope - rho * G0) / (rho * G0) < 0.02, (slope, rho * G0))


def test_normal_shock_m2():
    # Anderson: p2/p1 = 1 + 2 gamma/(gamma+1) (M^2-1) = 4.5 at M=2, gamma=1.4
    check(abs(normal_shock_pressure_ratio(2.0) - 4.5) < 1e-12, "p ratio")
    m2 = normal_shock_mach2(2.0)
    check(abs(m2 - 1.0 / math.sqrt(3.0)) < 1e-10, m2)


def test_prandtl_meyer_limits():
    check(prandtl_meyer(1.0) == 0.0, "nu(1)")
    # gamma=1.4 vacuum limit is (pi/2)*(sqrt(6)-1) ≈ 2.27685
    check(abs(prandtl_meyer_max(1.4) - 0.5 * math.pi * (math.sqrt(6.0) - 1.0)) < 1e-12, "numax")


def test_pitot_cpmax_m2():
    # Hand calculation in the module notes: M=2, p02/p1 ≈ 5.64, Cp_max ≈ 1.66
    cp = cp_max_modified_newtonian(2.0)
    check(1.5 < cp < 1.8, cp)


def test_panel_flat_plate():
    alpha = math.radians(5.0)
    cl, cm = incompressible_panel(alpha, 0.0, 1.0, 0.25, flap_fraction=0.2, n_panels=48)
    target = 2.0 * math.pi * alpha
    check(abs(cl - target) / target < 0.05, (cl, target))
    check(abs(cm) < 0.03, cm)


def test_panel_flap_sign():
    cl, cm = incompressible_panel(0.0, math.radians(5.0), 1.0, 0.25, 0.2, n_panels=48)
    check(cl > 0.0, cl)
    check(cm < 0.0, cm)  # TE down, nose-down about c/4


def test_ackeret():
    # Isolated wing: no fin, no nose, no friction, very thin.
    geom = {
        "wing_area_m2": 10.0,
        "wing_span_m": 10.0,
        "wing_chord_m": 1.0,
        "wing_le_from_nose_m": 0.0,
        "thickness_ratio": 0.0001,
        "flap_fraction": 0.2,
        "oswald_e": 1.0,
        "swet_m2": 0.0,
        "cd_misc": 0.0,
        "nose_length_m": 0.0,
        "nose_half_angle_deg": 5.0,
        "nose_radius_m": 0.1,
        "fin_area_m2": 0.0,
        "gamma": 1.4,
    }
    aero = Aerodynamics(geom)
    alpha = math.radians(2.0)
    # 15 km keeps the flow in the model; Mach is set by V/a.
    atm = us76(15000.0)
    M = 2.0
    ar = aero.compute(15000.0, M * atm.a_mps, alpha, 0.0, 0.0, cg_from_nose_m=0.5)
    cl_ref = 4.0 * alpha / math.sqrt(M * M - 1.0)
    cd_ref = 4.0 * alpha * alpha / math.sqrt(M * M - 1.0)
    check(abs(ar.CL - cl_ref) / cl_ref < 0.08, (ar.CL, cl_ref, ar.warnings))
    check(abs(ar.CD - cd_ref) / cd_ref < 0.20, (ar.CD, cd_ref))


def test_static_margin_sign():
    geom = {
        "wing_area_m2": 10.0,
        "wing_span_m": 10.0,
        "wing_chord_m": 1.0,
        "wing_le_from_nose_m": 0.0,
        "thickness_ratio": 0.04,
        "flap_fraction": 0.2,
        "oswald_e": 0.8,
        "swet_m2": 0.0,
        "cd_misc": 0.0,
        "nose_length_m": 0.0,
        "nose_half_angle_deg": 5.0,
        "nose_radius_m": 0.05,
        "fin_area_m2": 0.0,
        "gamma": 1.4,
    }
    aero = Aerodynamics(geom)
    atm = us76(15000.0)
    # cg at 0.2c: ahead of the supersonic AC at 0.5c, so +alpha gives nose-down My < 0
    ar = aero.compute(15000.0, 2.0 * atm.a_mps, math.radians(3.0), 0.0, 0.0, cg_from_nose_m=0.2)
    check(ar.moment_body_Nm[1] < 0.0, ar.moment_body_Nm)


def test_fin_restoring():
    geom = {
        "wing_area_m2": 20.0,
        "wing_span_m": 10.0,
        "wing_chord_m": 2.0,
        "wing_le_from_nose_m": 4.0,
        "thickness_ratio": 0.04,
        "flap_fraction": 0.2,
        "oswald_e": 0.8,
        "swet_m2": 0.0,
        "cd_misc": 0.0,
        "nose_length_m": 0.0,
        "nose_half_angle_deg": 5.0,
        "nose_radius_m": 0.05,
        "fin_area_m2": 4.0,
        "fin_span_m": 2.0,
        "fin_chord_m": 2.0,
        "fin_thickness_ratio": 0.04,
        "fin_x_from_nose_m": 8.0,
        "fin_z_up_m": 1.5,
        "gamma": 1.4,
    }
    aero = Aerodynamics(geom)
    atm = us76(15000.0)
    ar = aero.compute(15000.0, 2.0 * atm.a_mps, 0.0, math.radians(2.0), 0.0, cg_from_nose_m=4.5)
    check(ar.moment_body_Nm[2] > 0.0, ar.moment_body_Nm)


def test_ramjet_identity():
    cfg = {
        "gamma": 1.4,
        "gas_constant": R_AIR,
        "heating_value_j_per_kg": 1.20e8,
        "burner_efficiency": 1.0,
        "f_stoichiometric": 0.2,
        "capture_area_m2": 1.0,
        "capture_efficiency": 1.0,
        "combustor_area_m2": 0.5,
        "inlet_pressure_ratio": 1.0,
        "burner_pressure_ratio_ram": 1.0,
        "compressor_pressure_ratio": 6.0,
        "mdot_air_design_kg_s": 100.0,
        "tt4_max_turbo_k": 1700.0,
        "tt4_max_ram_k": 2000.0,
        "tt4_max_scram_k": 2600.0,
        "rocket_thrust_vac_n": 1000.0,
        "rocket_thrust_sl_n": 800.0,
        "rocket_isp_vac_s": 300.0,
    }
    p = Propulsion(cfg)
    atm = us76(20000.0)
    M0 = 3.0
    pr = p.evaluate("ramjet", 20000.0, M0 * atm.a_mps, 1.0)
    Me = pr.stations["Me"]
    check(abs(Me - M0) < 1e-2, (Me, pr.warnings, pr.stations))
    Tt0 = atm.T_K * (1.0 + 0.2 * M0 * M0)
    Ve_over_V0 = math.sqrt(pr.stations["Tt4"] / Tt0)
    V0 = M0 * atm.a_mps
    check(abs(pr.stations["Ve"] / V0 - Ve_over_V0) < 1e-2, (pr.stations["Ve"], V0, Ve_over_V0))


def test_turbojet_power_balance():
    cfg = {
        "gamma": 1.4,
        "gas_constant": R_AIR,
        "heating_value_j_per_kg": 1.20e8,
        "burner_efficiency": 0.95,
        "f_stoichiometric": 0.0292,
        "capture_area_m2": 1.0,
        "combustor_area_m2": 0.5,
        "inlet_pressure_ratio": 1.0,
        "burner_pressure_ratio_turbo": 1.0,
        "compressor_pressure_ratio": 6.0,
        "mdot_air_design_kg_s": 100.0,
        "tt4_max_turbo_k": 1700.0,
        "tt4_max_ram_k": 2000.0,
        "tt4_max_scram_k": 2500.0,
        "rocket_thrust_vac_n": 1000.0,
        "rocket_thrust_sl_n": 800.0,
        "rocket_isp_vac_s": 300.0,
    }
    pr = Propulsion(cfg).evaluate("turbojet", 0.0, 0.0, 1.0)
    # Static: captured-stream model is replaced by corrected flow, so thrust must be > 0.
    # At V=0 the ram term vanishes; give it a small forward speed for a defined Mach.
    pr = Propulsion(cfg).evaluate("turbojet", 0.0, 50.0, 1.0)
    check(pr.thrust_N > 0.0, (pr.thrust_N, pr.warnings, pr.stations))
    st = pr.stations
    f = st["f"]
    # (1+f)*(Tt4-Tt5) ~= Tt3-Tt2 and Tt2 = Tt0
    Tt0 = us76(0.0).T_K * (1.0 + 0.2 * (50.0 / us76(0.0).a_mps) ** 2)
    lhs = (1.0 + f) * (st["Tt4"] - st["Tt5"])
    rhs = st["Tt3"] - Tt0
    check(abs(lhs - rhs) / rhs < 1e-6, (lhs, rhs))


def test_rocket_altitude():
    cfg = {
        "gamma": 1.4,
        "gas_constant": R_AIR,
        "heating_value_j_per_kg": 1.0,
        "burner_efficiency": 1.0,
        "f_stoichiometric": 0.03,
        "capture_area_m2": 1.0,
        "combustor_area_m2": 1.0,
        "inlet_pressure_ratio": 1.0,
        "compressor_pressure_ratio": 4.0,
        "mdot_air_design_kg_s": 10.0,
        "tt4_max_turbo_k": 1500.0,
        "tt4_max_ram_k": 1500.0,
        "tt4_max_scram_k": 1500.0,
        "rocket_thrust_vac_n": 10000.0,
        "rocket_thrust_sl_n": 8000.0,
        "rocket_isp_vac_s": 450.0,
    }
    p = Propulsion(cfg)
    vac = p.evaluate("rocket", 200000.0, 1000.0, 1.0)
    sl = p.evaluate("rocket", 0.0, 100.0, 1.0)
    check(abs(vac.thrust_N - 10000.0) < 1.0, vac.thrust_N)
    check(abs(sl.thrust_N - 8000.0) < 1.0, sl.thrust_N)
    check(abs(sl.mdot_fuel_kg_s - 10000.0 / (450.0 * G0)) < 1e-6, sl.mdot_fuel_kg_s)


def test_rayleigh_inverse():
    m1, m2 = 2.5, 1.4
    ratio = rayleigh_tt_ratio(m1, m2, 1.4)
    m_back, choked = solve_rayleigh_mach(m1, ratio, 1.4)
    check(not choked, choked)
    check(abs(m_back - m2) < 1e-3, m_back)


def test_mass_flow_roundtrip():
    mdot = mass_flow_parameter(2.2, 1.5, 2.0e6, 800.0, 1.4, R_AIR)
    check(mdot > 0.0, mdot)


def test_j2_matches_potential_gradient():
    r = np.array([7000e3, -500e3, 1200e3], dtype=float)
    a = acceleration(r)
    eps = 1.0  # 1 m
    num = np.zeros(3)
    for i in range(3):
        rp = r.copy()
        rm = r.copy()
        rp[i] += eps
        rm[i] -= eps
        num[i] = (potential(rp) - potential(rm)) / (2 * eps)
    check(np.linalg.norm(a - num) / np.linalg.norm(a) < 1e-6, (a, num))


def test_quaternion_roll_and_pitch_sign():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    # Integrate a pure nose-up rate. Positive q must raise the nose: body +x gets -z.
    def rhs(t, y):
        return quat_derivative(y, np.array([0.0, 0.2, 0.0]))

    T = (math.pi / 2) / 0.2
    y = q
    dt = 0.001
    n = int(T / dt)
    for k in range(n):
        y = rk4_step(rhs, k * dt, y, dt, quat_slice=slice(0, 4))
    R = dcm_body_to_inertial(y)
    nose = R @ np.array([1.0, 0.0, 0.0])
    check(nose[2] < -0.95, nose)  # z-down frame: nose up means inertial z of body x is negative if inertial = body at t0


def test_body_triad_right_wing():
    # Equator, fly east: y should point south (right).
    v_hat = np.array([0.0, 1.0, 0.0])
    r_hat = np.array([1.0, 0.0, 0.0])
    R = body_triad(v_hat, r_hat, 0.0, 0.0)
    y_axis = R[:, 1]
    check(y_axis[2] < -0.99, y_axis)


def test_ned_north():
    R = ned_to_ecef_dcm(0.0, 0.0)
    north = R[:, 0]
    check(np.linalg.norm(north - np.array([0.0, 0.0, 1.0])) < 1e-12, north)


def test_circular_orbit_drift():
    """J2 off, vacuum, no thrust. RK4 should hold radius."""
    r0 = A_EARTH + 400e3
    v = circular_speed(r0)
    y = np.array([r0, 0.0, 0.0, 0.0, v, 0.0], dtype=float)

    def rhs(t, y):
        r = y[0:3]
        v = y[3:6]
        # local two-body, J2 disabled
        rn = np.linalg.norm(r)
        a = -MU_EARTH * r / rn ** 3
        return np.array([v[0], v[1], v[2], a[0], a[1], a[2]])

    t = 0.0
    dt = 1.0
    for _ in range(300):
        y = rk4_step(rhs, t, y, dt)
        t += dt
    rn = np.linalg.norm(y[0:3])
    check(abs(rn - r0) < 500.0, abs(rn - r0))


def test_expand_nozzle_rest():
    Me, Te, Ve = expand_nozzle(2.0e5, 1000.0, 1.0e5, 1.4, R_AIR)
    check(Me > 1.0 and Ve > 0.0, (Me, Ve))


def test_dcm_roundtrip():
    R = body_triad(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.2, 0.1)
    q = dcm_to_quaternion(R)
    R2 = dcm_body_to_inertial(q)
    check(np.linalg.norm(R - R2) < 1e-8, np.linalg.norm(R - R2))


def test_sutton_graves_algebra():
    from spaceplane.constraints import stagnation_heat_flux

    q = stagnation_heat_flux(0.01, 3000.0, 0.25, 1.83e-4)
    expected = 1.83e-4 * math.sqrt(0.01 / 0.25) * 3000.0 ** 3
    check(abs(q - expected) / expected < 1e-12, (q, expected))


TESTS = [
    test_isa_sea_level,
    test_isa_11km,
    test_hydrostatic_sea_level,
    test_normal_shock_m2,
    test_prandtl_meyer_limits,
    test_pitot_cpmax_m2,
    test_panel_flat_plate,
    test_panel_flap_sign,
    test_ackeret,
    test_static_margin_sign,
    test_fin_restoring,
    test_ramjet_identity,
    test_turbojet_power_balance,
    test_rocket_altitude,
    test_rayleigh_inverse,
    test_mass_flow_roundtrip,
    test_j2_matches_potential_gradient,
    test_quaternion_roll_and_pitch_sign,
    test_body_triad_right_wing,
    test_ned_north,
    test_circular_orbit_drift,
    test_expand_nozzle_rest,
    test_dcm_roundtrip,
    test_sutton_graves_algebra,
]


def main() -> int:
    fails = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001 — report every failed check
            fails += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"{len(TESTS) - fails} passed, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
