"""Propulsion.

All airbreathing cycles are calorically perfect with a single gamma. That is
the ideal-cycle model in Mattingly and in Heiser & Pratt introductory
chapters. It is not a real-gas combustor model. A warning is attached when
Tt4 > 1000 K.

Burner energy balance (Hill and Peterson; Mattingly), eta_b on the heating value:
    (1+f) cp Tt_out = cp Tt_in + f eta_b h_PR
    f = cp (Tt_out - Tt_in) / (eta_b h_PR - cp Tt_out)
f is not allowed to exceed the user stoichiometric limit.

Turbojet (Mattingly ideal cycle plus corrected flow):
    mdot * sqrt(Tt2) / pt2  =  mdot_ref * sqrt(T_ref) / p_ref
    Compressor power = turbine power, including fuel mass in the turbine stream.
    Nozzle expands isentropically to p0 (perfect expansion). pe = p0, so the
    pressure term in the thrust equation is zero.
    Inlet total pressure ratio and burner total pressure ratio are user inputs.
    Default 1 and 0.95 are not a measured map.

Ideal ramjet: same total-pressure construction with pi_c = 1 and no turbine.
    With pi_d = pi_b = 1 this gives the textbook identity Me = M0 and
    Ve = V0 * sqrt(Tt4/Tt0) (Heiser & Pratt; Mattingly).

Scramjet: constant-area Rayleigh heat addition (Anderson, Modern Compressible
Flow) downstream of a user area ratio A_combustor and a user inlet total
pressure ratio. Supersonic root only. If the captured mass flow exceeds the
choked mass flow of A_combustor, the result is flagged unstarted and thrust
is zero. This is a one-dimensional flag, not an inlet-unstart simulation.
Streamtube thrust at perfect expansion:
    F = mdot0 * ((1+f) Ve - V0)

Rocket (Sutton):
    Ae = (F_vac - F_sl) / p_sl
    F(h) = F_vac - pa(h) * Ae     at full throttle
    F(h, tau) = tau * F(h)
    mdot = tau * F_vac / (Isp_vac * g0)
Linear throttle of the altitude thrust does not model nozzle separation.

Overall efficiency reported as eta_0 = F * V0 / (mdot_fuel * h_PR) (Mattingly).
TSFC = mdot_fuel / F.
"""

from __future__ import annotations

import math

from .atmosphere import us76
from .constants import G0, GAMMA_AIR, P0_STD, R_AIR, T0_STD
from .types import PropulsionResult


def _cp(gamma: float, R: float) -> float:
    return gamma * R / (gamma - 1.0)


def _result(thrust, mdot_f, mdot_a, mode, Me, stations, notes) -> PropulsionResult:
    isp = thrust / (mdot_f * G0) if mdot_f > 1e-8 and thrust > 0 else 0.0
    tsfc = mdot_f / thrust if thrust > 1.0 else float("nan")
    return PropulsionResult(thrust, mdot_f, mdot_a, isp, tsfc, mode, Me, stations, notes)


def fuel_air_ratio(cp: float, Tt_in: float, Tt_out: float, eta_b: float, h_pr: float, f_max: float) -> tuple[float, float, str | None]:
    if Tt_out <= Tt_in + 1e-6:
        return 0.0, Tt_in, None
    denom = eta_b * h_pr - cp * Tt_out
    if denom <= 0.0:
        # Temperature is thermodynamically unreachable at this heating value.
        Tt_out = 0.5 * (Tt_in + eta_b * h_pr / cp)
        denom = eta_b * h_pr - cp * Tt_out
    f = cp * (Tt_out - Tt_in) / denom
    note = None
    if f > f_max:
        f = f_max
        Tt_out = (cp * Tt_in + f * eta_b * h_pr) / ((1.0 + f) * cp)
        note = "fuel-air ratio capped at the user stoichiometric limit; Tt4 reduced"
    if f < 0.0:
        return 0.0, Tt_in, "negative fuel flow suppressed"
    return f, Tt_out, note


def expand_nozzle(pt: float, Tt: float, p_back: float, gamma: float, R: float) -> tuple[float, float, float]:
    if pt <= p_back * 1.0000001 or Tt <= 0.0:
        return 0.0, Tt, 0.0
    expo = (gamma - 1.0) / gamma
    Me2 = 2.0 / (gamma - 1.0) * ((pt / p_back) ** expo - 1.0)
    Me = math.sqrt(max(Me2, 0.0))
    Te = Tt / (1.0 + 0.5 * (gamma - 1.0) * Me2)
    Ve = Me * math.sqrt(max(gamma * R * Te, 0.0))
    return Me, Te, Ve


def mass_flow_parameter(mach: float, area: float, pt: float, Tt: float, gamma: float, R: float) -> float:
    """mdot = A pt/sqrt(Tt) * sqrt(gamma/R) * M * (1+(gamma-1)/2 M^2)^(-(gamma+1)/(2(gamma-1)))"""
    if mach <= 0.0 or area <= 0.0 or pt <= 0.0 or Tt <= 0.0:
        return 0.0
    expo = -0.5 * (gamma + 1.0) / (gamma - 1.0)
    return (
        area
        * pt
        / math.sqrt(Tt)
        * math.sqrt(gamma / R)
        * mach
        * (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** expo
    )


def rayleigh_tt_ratio(m1: float, m2: float, gamma: float) -> float:
    """Tt2/Tt1 for Rayleigh flow (Anderson)."""
    def g(m):
        return m * m * (1.0 + gamma * m1 * m1) ** 2  # placeholder replaced below

    a1 = 1.0 + gamma * m1 * m1
    a2 = 1.0 + gamma * m2 * m2
    b1 = 1.0 + 0.5 * (gamma - 1.0) * m1 * m1
    b2 = 1.0 + 0.5 * (gamma - 1.0) * m2 * m2
    return (m2 * m2 / (m1 * m1)) * (a1 * a1) / (a2 * a2) * (b2 / b1)


def rayleigh_pt_ratio(m1: float, m2: float, gamma: float) -> float:
    a1 = 1.0 + gamma * m1 * m1
    a2 = 1.0 + gamma * m2 * m2
    b1 = 1.0 + 0.5 * (gamma - 1.0) * m1 * m1
    b2 = 1.0 + 0.5 * (gamma - 1.0) * m2 * m2
    return (a1 / a2) * (b2 / b1) ** (gamma / (gamma - 1.0))


def solve_rayleigh_mach(m1: float, tt_ratio: float, gamma: float) -> tuple[float, bool]:
    """Find M2 in (1, M1) for a requested Tt2/Tt1. Second flag is True if choked."""
    if m1 <= 1.01:
        return m1, True
    tt_choke = rayleigh_tt_ratio(m1, 1.0001, gamma)
    if tt_ratio >= tt_choke:
        return 1.0001, True
    if tt_ratio <= 1.0:
        return m1, False
    lo, hi = 1.0001, m1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rayleigh_tt_ratio(m1, mid, gamma) > tt_ratio:
            # Closer to 1 raises Tt ratio, so mid is too close to 1 (ratio too high): move up
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


class Propulsion:
    def __init__(self, cfg: dict):
        self.c = cfg
        self.gamma = float(cfg.get("gamma", GAMMA_AIR))
        self.R = float(cfg.get("gas_constant", R_AIR))
        self.cp = _cp(self.gamma, self.R)

    def evaluate(self, mode: str, h_m: float, airspeed: float, throttle: float) -> PropulsionResult:
        throttle = float(max(0.0, min(1.0, throttle)))
        mode = mode.lower().strip()
        if mode in ("rocket",):
            return self._rocket(h_m, airspeed, throttle)
        if mode in ("turbojet", "ramjet", "scramjet"):
            return self._airbreather(mode, h_m, airspeed, throttle)
        return _result(0.0, 0.0, 0.0, "none", 0.0, {}, [])

    def _airbreather(self, mode: str, h_m: float, V0: float, throttle: float) -> PropulsionResult:
        notes = []
        atm = us76(h_m)
        if atm.rho <= 1e-6 or V0 < 1.0:
            notes.append("no captured airflow")
            return _result(0.0, 0.0, 0.0, mode, 0.0, {}, notes)
        g = self.gamma
        R = self.R
        cp = self.cp
        M0 = V0 / atm.a_mps
        T0 = atm.T_K
        p0 = atm.p_Pa
        tau_r = 1.0 + 0.5 * (g - 1.0) * M0 * M0
        Tt0 = T0 * tau_r
        pt0 = p0 * tau_r ** (g / (g - 1.0))
        if Tt0 > 1000.0 or float(self.c.get("tt4_max_turbo_k", 0)) > 1000.0:
            notes.append("calorically perfect gamma is outside the low-temperature air range; cycle is qualitative")
        eta_b = float(self.c.get("burner_efficiency", 1.0))
        h_pr = float(self.c["heating_value_j_per_kg"])
        f_max = float(self.c.get("f_stoichiometric", 0.0292))
        pi_d = float(self.c.get("inlet_pressure_ratio", 1.0))
        if M0 > 1.0 and abs(pi_d - 1.0) < 1e-9:
            notes.append("inlet total-pressure ratio is 1: ideal inlet, not a measured recovery")

        if mode == "scramjet":
            return self._scramjet(M0, V0, T0, p0, atm.rho, Tt0, pt0, pi_d, throttle, eta_b, h_pr, f_max, notes)

        # Turbojet or ramjet Brayton model
        if mode == "turbojet":
            pi_c = float(self.c.get("compressor_pressure_ratio", 8.0))
            Tt4_max = float(self.c["tt4_max_turbo_k"])
            pi_b = float(self.c.get("burner_pressure_ratio_turbo", 0.95))
            mdot_ref = float(self.c["mdot_air_design_kg_s"])
        else:
            pi_c = 1.0
            Tt4_max = float(self.c["tt4_max_ram_k"])
            pi_b = float(self.c.get("burner_pressure_ratio_ram", 1.0))
            mdot_ref = None

        pt2 = pi_d * pt0
        Tt2 = Tt0
        if mode == "turbojet":
            mdot_a = mdot_ref * (pt2 / P0_STD) * math.sqrt(T0_STD / Tt2)
        else:
            eta_c = float(self.c.get("capture_efficiency", 1.0))
            mdot_a = atm.rho * V0 * float(self.c["capture_area_m2"]) * eta_c
        if mdot_a <= 0.0 or throttle <= 0.0:
            return _result(0.0, 0.0, mdot_a, mode, 0.0, {"M0": M0}, notes)

        tau_c = pi_c ** ((g - 1.0) / g)
        Tt3 = Tt2 * tau_c
        pt3 = pt2 * pi_c
        Tt4_cmd = Tt3 + throttle * max(Tt4_max - Tt3, 0.0)
        f, Tt4, fnote = fuel_air_ratio(cp, Tt3, Tt4_cmd, eta_b, h_pr, f_max)
        if fnote:
            notes.append(fnote)
        pt4 = pt3 * pi_b
        if mode == "turbojet":
            # (1+f) cp (Tt4 - Tt5) = cp (Tt3 - Tt2)
            power = (Tt3 - Tt2) / max(1.0 + f, 1e-8)
            Tt5 = Tt4 - power
            if Tt5 <= T0:
                notes.append("turbine cannot drive the compressor at this flight condition")
                return _result(0.0, 0.0, mdot_a, mode, 0.0, {"Tt3": Tt3, "Tt4": Tt4}, notes)
            pt5 = pt4 * (Tt5 / Tt4) ** (g / (g - 1.0))
        else:
            Tt5 = Tt4
            pt5 = pt4
        Me, Te, Ve = expand_nozzle(pt5, Tt5, p0, g, R)
        thrust = mdot_a * ((1.0 + f) * Ve - V0)
        mdot_f = f * mdot_a
        stations = {
            "M0": M0,
            "Tt0": Tt0,
            "Tt3": Tt3,
            "Tt4": Tt4,
            "Tt5": Tt5,
            "Me": Me,
            "Ve": Ve,
            "f": f,
            "mdot_air": mdot_a,
        }
        if f > 0.0 and V0 > 0.0 and thrust > 0.0:
            stations["eta_overall"] = thrust * V0 / (mdot_f * h_pr)
        return _result(thrust, mdot_f, mdot_a, mode, Me, stations, notes)

    def _scramjet(self, M0, V0, T0, p0, rho, Tt0, pt0, pi_d, throttle, eta_b, h_pr, f_max, notes):
        g = self.gamma
        R = self.R
        cp = self.cp
        if M0 < 1.5:
            notes.append("scramjet model rejected below Mach 1.5")
            return _result(0.0, 0.0, 0.0, "scramjet", 0.0, {}, notes)
        A0 = float(self.c["capture_area_m2"])
        A2 = float(self.c["combustor_area_m2"])
        eta_c = float(self.c.get("capture_efficiency", 1.0))
        mdot0 = rho * V0 * A0 * eta_c
        pt2 = pi_d * pt0
        Tt2 = Tt0
        mdot_choke = mass_flow_parameter(1.0, A2, pt2, Tt2, g, R)
        if mdot0 > mdot_choke * 1.001:
            notes.append("combustor area cannot pass the captured mass flow (choked); thrust set to 0")
            return _result(0.0, 0.0, mdot0, "scramjet", 0.0, {"unstart": True}, notes)
        # Supersonic root: mdot decreases as M increases above 1.
        lo, hi = 1.01, max(M0, 1.2)
        # Extend hi until mdot(hi) <= mdot0
        for _ in range(20):
            if mass_flow_parameter(hi, A2, pt2, Tt2, g, R) <= mdot0:
                break
            hi *= 1.4
        if mass_flow_parameter(hi, A2, pt2, Tt2, g, R) > mdot0:
            notes.append("no supersonic isolator solution")
            return _result(0.0, 0.0, mdot0, "scramjet", 0.0, {}, notes)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mass_flow_parameter(mid, A2, pt2, Tt2, g, R) > mdot0:
                lo = mid  # still too much flow, need higher Mach
            else:
                hi = mid
        M2 = 0.5 * (lo + hi)
        if M2 <= 1.05:
            notes.append("isolator Mach is not supersonic")
            return _result(0.0, 0.0, mdot0, "scramjet", 0.0, {"M2": M2}, notes)
        Tt4_max = float(self.c["tt4_max_scram_k"])
        Tt_cmd = Tt2 + throttle * max(Tt4_max - Tt2, 0.0)
        # Rayleigh limit
        tt_choke = rayleigh_tt_ratio(M2, 1.0001, g)
        Tt_choke = Tt2 * tt_choke
        choked = False
        if Tt_cmd > Tt_choke:
            Tt_cmd = Tt_choke
            choked = True
            notes.append("Rayleigh line reached thermal choking; fuel reduced")
        f, Tt4, fnote = fuel_air_ratio(cp, Tt2, Tt_cmd, eta_b, h_pr, f_max)
        if fnote:
            notes.append(fnote)
        if Tt2 <= 0:
            return _result(0.0, 0.0, mdot0, "scramjet", 0.0, {}, notes)
        M4, choke_flag = solve_rayleigh_mach(M2, Tt4 / Tt2, g)
        if choke_flag:
            notes.append("combustor solved at the choke point")
        pt4 = pt2 * rayleigh_pt_ratio(M2, M4, g)
        Me, Te, Ve = expand_nozzle(pt4, Tt4, p0, g, R)
        thrust = mdot0 * ((1.0 + f) * Ve - V0)
        stations = {"M0": M0, "M2": M2, "M4": M4, "Me": Me, "Tt4": Tt4, "f": f, "choked": choked or choke_flag, "Ve": Ve}
        if f > 0 and thrust > 0:
            stations["eta_overall"] = thrust * V0 / (f * mdot0 * h_pr)
        return _result(thrust, f * mdot0, mdot0, "scramjet", Me, stations, notes)

    def _rocket(self, h_m: float, airspeed: float, throttle: float) -> PropulsionResult:
        atm = us76(max(h_m, 0.0))
        # Above the US76 model, static pressure is ~0. Use 0 so the nozzle is at vacuum.
        pa = atm.p_Pa if atm.in_model else 0.0
        F_vac = float(self.c["rocket_thrust_vac_n"])
        F_sl = float(self.c["rocket_thrust_sl_n"])
        Isp = float(self.c["rocket_isp_vac_s"])
        if F_vac <= F_sl:
            raise ValueError("rocket_thrust_vac_n must exceed rocket_thrust_sl_n so Ae > 0")
        Ae = (F_vac - F_sl) / P0_STD
        F_full = F_vac - pa * Ae
        thrust = throttle * F_full
        mdot = throttle * F_vac / (Isp * G0)
        notes = ["rocket: F = throttle * (F_vac - pa*Ae); separation not modeled"]
        if thrust < 0.0:
            notes.append("altitude thrust went negative; clamped to 0")
            thrust = 0.0
        return _result(thrust, mdot, 0.0, "rocket", float("nan"), {"Ae_m2": Ae, "F_full": F_full, "pa": pa}, notes)
