"""Aerodynamics.

Subsonic (M <= 0.85)
    2D incompressible 1/4-3/4 vortex panel (Katz and Plotkin; Glauert thin
    airfoil). Prandtl-Glauert: cl(M) = cl_inc / sqrt(1-M^2).
    Finite wing (Anderson):
        CL = cl / (1 + 2 / (beta AR))
    which is a = a0 / (beta + a0/(pi AR)) with a0 = 2 pi.
    Induced drag: CD = CL^2 / (pi AR e), e is a user Oswald factor.
    Symmetric section AC at c/4. Flap is a real deflection of the mean line
    (TE down is negative z_up), not a made-up Cm_delta.

Supersonic (M >= 1.25)
    Shock-expansion on a symmetric double wedge (Anderson, Hypersonic and
    High-Temperature Gas Dynamics; Modern Compressible Flow).
    Wedge half-angle epsilon = atan(t/c) because max thickness is at midchord:
    tan(epsilon) = (t/2)/(c/2) = t/c.
    Strip theory: 2D section applied over the planform. No sweep, no tip
    correction. High-AR unswept wings are the validity region. Low AR
    overpredicts lift; replace with a CFD or test table for a real vehicle.
    Supersonic AC of a symmetric section is at c/2 (Anderson).

Transonic
    Linear blend in Mach of the two force/moment vectors from 0.85 to 1.25.
    Not a physical theory. Drag rise is not predicted.

Nose, M >= 3
    Modified Newtonian local inclination (Lees; Anderson):
        Cp = Cp_max * max(n_hat · v_hat, 0)^2
    Cp_max from the Rayleigh-pitot pressure (normal shock then isentropic
    stagnation) divided by (0.5 gamma M^2). Leeward panels contribute 0.
    Tangent geometry is a sharp cone. Bluntness drag is not predicted; put it
    in cd_misc if a test value exists. Rn is used only for heating.

Skin friction
    Eckert reference temperature as quoted by Anderson:
        T*/Te = 1 + 0.032 M^2 + 0.58 (Tw/Te - 1)
    Adiabatic wall, recovery factor r = Pr^(1/3), Pr = 0.72 (user input).
    Laminar: Cf = 1.328 / sqrt(Re*)  (Blasius mean).
    Turbulent: Cf = 0.455 / (log10 Re*)^2.58  (Schlichting).
    Transition Reynolds number is prescribed, not predicted.
    Form factor defaults to 1. No Hoerner/Raymer form-factor curve fit is
    applied, because those coefficients are not re-derived here.

cd_misc
    User incremental drag (base, inlet, interference) added as a force
    opposing the air-relative velocity. Default in the example is illustrative.
"""

from __future__ import annotations

import math

import numpy as np

from .atmosphere import us76
from .constants import GAMMA_AIR, R_AIR
from .frames import decompose_aero_force, force_from_coefficients
from .types import AeroResult


def _zeros(alpha: float, beta: float, mach: float, notes: list[str]) -> AeroResult:
    z = np.zeros(3)
    return AeroResult(z, z.copy(), 0, 0, 0, 0, 0, 0, alpha, beta, mach, 0, 0, 0, 0, 0, notes)


def prandtl_meyer(mach: float, gamma: float = GAMMA_AIR) -> float:
    if mach <= 1.0:
        return 0.0
    gp = gamma + 1.0
    gm = gamma - 1.0
    return math.sqrt(gp / gm) * math.atan(math.sqrt(gm / gp * (mach * mach - 1.0))) - math.atan(math.sqrt(mach * mach - 1.0))


def prandtl_meyer_max(gamma: float = GAMMA_AIR) -> float:
    return 0.5 * math.pi * (math.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)


def inverse_prandtl_meyer(nu: float, gamma: float = GAMMA_AIR) -> float:
    if nu <= 0.0:
        return 1.0
    nu_max = prandtl_meyer_max(gamma) - 1e-8
    if nu >= nu_max:
        return 50.0
    lo, hi = 1.0000001, 50.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if prandtl_meyer(mid, gamma) < nu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_shock_pressure_ratio(mach: float, gamma: float = GAMMA_AIR) -> float:
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (mach * mach - 1.0)


def normal_shock_mach2(mach: float, gamma: float = GAMMA_AIR) -> float:
    num = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    den = gamma * mach * mach - 0.5 * (gamma - 1.0)
    return math.sqrt(num / den)


def pitot_pressure_ratio(mach: float, gamma: float = GAMMA_AIR) -> float:
    """p02/p1. Subsonic: isentropic. Supersonic: normal shock then isentropic."""
    if mach < 1.0:
        return (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** (gamma / (gamma - 1.0))
    p2_p1 = normal_shock_pressure_ratio(mach, gamma)
    m2 = normal_shock_mach2(mach, gamma)
    p02_p2 = (1.0 + 0.5 * (gamma - 1.0) * m2 * m2) ** (gamma / (gamma - 1.0))
    return p2_p1 * p02_p2


def cp_max_modified_newtonian(mach: float, gamma: float = GAMMA_AIR) -> float:
    if mach < 0.05:
        return 0.0
    q_over_p = 0.5 * gamma * mach * mach
    return (pitot_pressure_ratio(mach, gamma) - 1.0) / q_over_p


def oblique_shock_beta(mach: float, theta: float, gamma: float = GAMMA_AIR) -> float | None:
    """Weak-shock wave angle. None if detached or if inputs are non-physical."""
    if mach <= 1.0 or theta <= 1e-8:
        return None
    mu = math.asin(min(1.0 - 1e-12, 1.0 / mach))

    def theta_of_beta(beta: float) -> float:
        sb = math.sin(beta)
        return math.atan(
            2.0 / math.tan(beta) * (mach * mach * sb * sb - 1.0)
            / (mach * mach * (gamma + math.cos(2.0 * beta)) + 2.0)
        )

    betas = np.linspace(mu + 1e-4, 0.5 * math.pi - 1e-4, 100)
    ths = np.array([theta_of_beta(float(b)) for b in betas])
    imax = int(np.argmax(ths))
    if theta > float(ths[imax]):
        return None
    lo, hi = float(betas[0]), float(betas[imax])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if theta_of_beta(mid) < theta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _isentropic_a(mach: float, gamma: float) -> float:
    return (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** (gamma / (gamma - 1.0))


def flow_deflect(mach: float, p: float, dtheta: float, gamma: float = GAMMA_AIR) -> tuple[float, float, str | None]:
    """dtheta > 0 is compression (oblique shock). dtheta < 0 is expansion."""
    if abs(dtheta) < 1e-10:
        return mach, p, None
    if dtheta > 0.0:
        if mach <= 1.01:
            return mach, p, "subsonic_compression_skipped"
        beta = oblique_shock_beta(mach, dtheta, gamma)
        if beta is None:
            p2 = p * normal_shock_pressure_ratio(mach, gamma)
            return normal_shock_mach2(mach, gamma), p2, "detached_shock"
        mn = mach * math.sin(beta)
        p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mn * mn - 1.0)
        mn2sq = (1.0 + 0.5 * (gamma - 1.0) * mn * mn) / (gamma * mn * mn - 0.5 * (gamma - 1.0))
        m2 = math.sqrt(max(mn2sq, 1e-8)) / max(math.sin(beta - dtheta), 1e-6)
        return m2, p * p_ratio, None
    # Expansion
    if mach <= 1.01:
        return mach, p, "subsonic_expansion_skipped"
    nu2 = prandtl_meyer(mach, gamma) + (-dtheta)
    if nu2 >= prandtl_meyer_max(gamma):
        return 50.0, p * 1e-6, "expansion_limited"
    m2 = inverse_prandtl_meyer(nu2, gamma)
    p2 = p * _isentropic_a(mach, gamma) / _isentropic_a(m2, gamma)
    return m2, p2, None


def incompressible_panel(
    alpha: float,
    delta_te_down: float,
    chord: float,
    x_cg_from_le: float,
    flap_fraction: float,
    n_panels: int = 48,
) -> tuple[float, float]:
    """Return (cl, cm_noseup about x_cg_from_le) at unit freestream, incompressible.

    x is aft from the leading edge, z is up. TE down is negative z.
    Circulation sign: positive Gamma produces +z force (see module math in frames
    and the self-test against 2 pi alpha and cm_c/4 ~ 0).
    """
    x_h = chord * (1.0 - flap_fraction)
    xs = np.linspace(0.0, chord, n_panels + 1)
    zs = np.zeros_like(xs)
    for i, x in enumerate(xs):
        if x > x_h:
            zs[i] = -(x - x_h) * math.tan(delta_te_down)
    x1, z1 = xs[:-1], zs[:-1]
    x2, z2 = xs[1:], zs[1:]
    dx, dz = x2 - x1, z2 - z1
    ds = np.hypot(dx, dz)
    tx, tz = dx / ds, dz / ds
    nx, nz = -tz, tx  # CCW, +z on a flat plate
    xv = x1 + 0.25 * dx
    zv = z1 + 0.25 * dz
    xc = x1 + 0.75 * dx
    zc = z1 + 0.75 * dz
    ca, sa = math.cos(alpha), math.sin(alpha)
    A = np.zeros((n_panels, n_panels))
    rhs = np.zeros(n_panels)
    for i in range(n_panels):
        rhs[i] = -(ca * nx[i] + sa * nz[i])
        for j in range(n_panels):
            rx = xc[i] - xv[j]
            rz = zc[i] - zv[j]
            r2 = rx * rx + rz * rz + 1e-18
            u = rz / (2.0 * math.pi * r2)
            w = -rx / (2.0 * math.pi * r2)
            A[i, j] = u * nx[i] + w * nz[i]
    gamma = np.linalg.solve(A, rhs)
    # Force on each lumped vortex, rho = V = 1:
    # F_aft = -Gamma sin(alpha), F_up = Gamma cos(alpha)
    cl_num = 0.0
    m_le = 0.0
    for j in range(n_panels):
        g = float(gamma[j])
        f_aft = -g * sa
        f_up = g * ca
        cl_num += f_up * ca - f_aft * sa  # lift per unit span, since L = F · u_L and u_L=(-sa, ca) in (aft, up)? 
        # In (x_aft, z_up), V_hat = (ca, sa), u_L_up_perp = (-sa, ca)
        # F · u_L = f_aft*(-sa) + f_up*ca. Yes that is lift per unit span.
        m_le += (0.0 - float(xv[j])) * f_up + float(zv[j]) * f_aft
    # The loop above accumulated lift incorrectly if I also want total Gamma form.
    # Use Kj: L' = rho V Gamma_total with this sign convention, cl = 2 Gamma / (V c).
    gamma_total = float(np.sum(gamma))
    cl = 2.0 * gamma_total / chord
    # Moment about cg. M_noseup = (x_ref - x) F_up + z F_aft, summed.
    m = 0.0
    for j in range(n_panels):
        g = float(gamma[j])
        f_aft = -g * sa
        f_up = g * ca
        m += (x_cg_from_le - float(xv[j])) * f_up + float(zv[j]) * f_aft
    cm = m / (0.5 * chord * chord)
    return cl, cm


def skin_friction_cf(mach: float, reynolds_edge: float, T_e: float, p_e: float, recovery_pr: float, transition_re: float) -> tuple[float, float]:
    """Return (Cf, Re_star) using Eckert T* and an incompressible Cf law at Re*."""
    if reynolds_edge < 1.0 or T_e <= 0.0:
        return 0.0, 0.0
    r = recovery_pr ** (1.0 / 3.0)
    T_aw = T_e * (1.0 + r * 0.5 * (GAMMA_AIR - 1.0) * mach * mach)
    # Adiabatic wall: Tw = T_aw. Anderson/Eckert T*.
    T_star = T_e * (1.0 + 0.032 * mach * mach + 0.58 * (T_aw / T_e - 1.0))
    mu_e = us76_mu_from_T(T_e)
    mu_star = us76_mu_from_T(T_star)
    # Re_edge = rho_e V L / mu_e, so rho_e V L = Re_edge * mu_e
    # Re* = rho* V L / mu* , rho* = p / (R T*)
    rho_e_V_L = reynolds_edge * mu_e
    # rho_e = p/(R Te), V L = rho_e_V_L / rho_e
    rho_e = p_e / (R_AIR * T_e)
    if rho_e <= 0.0:
        return 0.0, 0.0
    VL = rho_e_V_L / rho_e
    rho_star = p_e / (R_AIR * T_star)
    re_star = rho_star * VL / mu_star
    if re_star < transition_re:
        cf = 1.328 / math.sqrt(max(re_star, 1.0))
    else:
        cf = 0.455 / (math.log10(re_star) ** 2.58)
    return cf, re_star


def us76_mu_from_T(T: float) -> float:
    from .constants import SUTHERLAND_B, SUTHERLAND_S

    return SUTHERLAND_B * T ** 1.5 / (T + SUTHERLAND_S)


class Aerodynamics:
    def __init__(self, geometry: dict, limits: dict | None = None):
        self.g = geometry
        self.limits = limits or {}
        self.gamma = float(geometry.get("gamma", GAMMA_AIR))

    def compute(
        self,
        h_m: float,
        airspeed: float,
        alpha: float,
        beta: float,
        elevator: float,
        cg_from_nose_m: float,
    ) -> AeroResult:
        notes: list[str] = []
        atm = us76(h_m)
        V = float(airspeed)
        if V < 1.0 or atm.rho <= 0.0 or not atm.in_model and atm.rho <= 0.0:
            if h_m > 86000.0:
                notes.append("above US76 lapse model: aero forces set to zero")
            return _zeros(alpha, beta, 0.0, notes)
        mach = V / atm.a_mps
        q = 0.5 * atm.rho * V * V
        S = float(self.g["wing_area_m2"])
        chord = float(self.g["wing_chord_m"])
        span = float(self.g["wing_span_m"])
        AR = span * span / S
        t_c = float(self.g["thickness_ratio"])
        le = float(self.g["wing_le_from_nose_m"])
        x_le_body = cg_from_nose_m - le  # body x, forward positive
        x_cg_from_le = cg_from_nose_m - le

        F = np.zeros(3)
        M = np.zeros(3)
        if mach <= 0.85:
            Fw, Mw, wnotes = self._subsonic_wing(mach, alpha, elevator, q, S, chord, AR, x_cg_from_le)
            notes.extend(wnotes)
            F += Fw
            M += Mw
        elif mach >= 1.25:
            Fw, Mw, wnotes = self._wedge_wing(mach, alpha, elevator, atm.p_Pa, q, S, chord, t_c, x_le_body)
            notes.extend(wnotes)
            F += Fw
            M += Mw
        else:
            # Numerical bridge only.
            notes.append("transonic bridge: linear blend, not a prediction of drag rise")
            F1, M1, n1 = self._subsonic_wing(0.85, alpha, elevator, q, S, chord, AR, x_cg_from_le)
            # Rebuild subsonic at the actual q but with M=0.85 coefficients already used.
            F2, M2, n2 = self._wedge_wing(1.25, alpha, elevator, atm.p_Pa, q, S, chord, t_c, x_le_body)
            notes.extend(n1)
            notes.extend(n2)
            w = (mach - 0.85) / (1.25 - 0.85)
            F += (1.0 - w) * F1 + w * F2
            M += (1.0 - w) * M1 + w * M2

        # Friction and user incremental drag. Moment of these is neglected.
        Lref = float(self.g.get("friction_length_m", chord))
        re_edge = atm.rho * V * Lref / max(atm.mu, 1e-12)
        pr = float(self.g.get("prandtl", 0.72))
        re_tr = float(self.g.get("transition_reynolds", 1.0e6))
        Cf, _ = skin_friction_cf(mach, re_edge, atm.T_K, atm.p_Pa, pr, re_tr)
        FF = float(self.g.get("form_factor", 1.0))
        Swet = float(self.g.get("swet_m2", 0.0))
        v_hat = self._vhat(alpha, beta)
        D_f = q * Cf * FF * Swet
        F += -D_f * v_hat
        D_misc = q * S * float(self.g.get("cd_misc", 0.0))
        F += -D_misc * v_hat

        if mach >= 3.0 and float(self.g.get("nose_length_m", 0.0)) > 0.0:
            Fc, Mc, n = self._newtonian_nose(mach, alpha, beta, q, cg_from_nose_m)
            notes.extend(n)
            F += Fc
            M += Mc

        if float(self.g.get("fin_area_m2", 0.0)) > 0.0:
            Ff, Mf, n = self._fin(mach, beta, elevator_unused=0.0, q=q, cg_from_nose_m=cg_from_nose_m, atm_p=atm.p_Pa)
            notes.extend(n)
            F += Ff
            M += Mf

        if abs(alpha) > math.radians(25.0) or abs(beta) > math.radians(20.0):
            notes.append("angle outside the intended envelope of the aero model")

        L, D, Y = decompose_aero_force(F, alpha, beta)
        CL = L / (q * S) if q * S > 0 else 0.0
        CD = D / (q * S) if q * S > 0 else 0.0
        CY = Y / (q * S) if q * S > 0 else 0.0
        Cm = float(M[1] / (q * S * chord)) if q * S * chord > 0 else 0.0
        b = span
        Cl = float(M[0] / (q * S * b)) if q * S * b > 0 else 0.0
        Cn = float(M[2] / (q * S * b)) if q * S * b > 0 else 0.0
        return AeroResult(F, M, CL, CD, CY, Cm, Cl, Cn, alpha, beta, mach, q, re_edge, Cf, L, D, notes)

    @staticmethod
    def _vhat(alpha: float, beta: float) -> np.ndarray:
        return np.array(
            [math.cos(alpha) * math.cos(beta), math.sin(beta), math.sin(alpha) * math.cos(beta)],
            dtype=float,
        )

    def _subsonic_wing(self, mach, alpha, elevator, q, S, chord, AR, x_cg_from_le):
        notes = []
        if mach >= 1.0:
            mach = 0.85
        beta_pg = math.sqrt(max(1.0 - mach * mach, 0.05))
        flap_frac = float(self.g.get("flap_fraction", 0.2))
        cl_inc, cm_inc = incompressible_panel(alpha, elevator, chord, x_cg_from_le, flap_frac)
        cl = cl_inc / beta_pg
        cm = cm_inc / beta_pg
        k = 1.0 / (1.0 + 2.0 / (beta_pg * max(AR, 0.5)))
        CL = k * cl
        Cm = k * cm
        e = float(self.g.get("oswald_e", 0.85))
        CD = CL * CL / (math.pi * AR * e)
        L = q * S * CL
        D = q * S * CD
        F = force_from_coefficients(L, D, 0.0, alpha, 0.0)
        M = np.array([0.0, Cm * q * S * chord, 0.0])
        notes.append("subsonic: panel + Prandtl-Glauert + finite-wing factor; Oswald e is a user input")
        return F, M, notes

    def _wedge_wing(self, mach, alpha, elevator, p_inf, q, S, chord, t_c, x_le_body):
        notes = []
        gamma = self.gamma
        eps = math.atan(max(t_c, 1e-6))
        # theta is the surface angle relative to the freestream, positive when
        # that side sees compression. delta > 0 is trailing edge down.
        d = elevator
        turns = {
            "lf": alpha + eps,
            "la": alpha - eps + d,
            "uf": -alpha + eps,
            "ua": -alpha - eps - d,
        }
        p_lf, n1 = self._side_pressures(mach, p_inf, gamma, turns["lf"], turns["la"])
        p_la = p_lf[1]
        p_lf = p_lf[0]
        p_uf, n2 = self._side_pressures(mach, p_inf, gamma, turns["uf"], turns["ua"])
        p_ua = p_uf[1]
        p_uf = p_uf[0]
        notes.extend(n1)
        notes.extend(n2)
        # Tangent angle of the aft-pointing surface above the chord.
        sig = {
            "lf": -eps,
            "la": eps - d,
            "uf": eps,
            "ua": -eps - d,
        }
        pressures = {"lf": p_lf, "la": p_la, "uf": p_uf, "ua": p_ua}
        x_aft = {"lf": 0.25 * chord, "la": 0.75 * chord, "uf": 0.25 * chord, "ua": 0.75 * chord}
        t = t_c * chord
        z_te = -(0.5 * chord) * math.sin(d)
        z_up = {
            "lf": -0.25 * t,
            "uf": 0.25 * t,
            "la": 0.5 * (-0.5 * t + z_te),
            "ua": 0.5 * (0.5 * t + z_te),
        }
        F = np.zeros(3)
        M = np.zeros(3)
        for key in ("lf", "la", "uf", "ua"):
            sigma = sig[key]
            A = (0.5 * S) / max(math.cos(abs(sigma)), 0.2)
            if key[0] == "l":
                n_out = np.array([math.sin(sigma), -math.cos(sigma)])
            else:
                n_out = np.array([-math.sin(sigma), math.cos(sigma)])
            dF_aft_up = -pressures[key] * A * n_out
            dF = np.array([-dF_aft_up[0], 0.0, -dF_aft_up[1]])
            r = np.array([x_le_body - x_aft[key], 0.0, -z_up[key]])
            F += dF
            M += np.cross(r, dF)
        notes.append("supersonic: shock-expansion double wedge, 2D strip theory, no sweep")
        return F, M, notes

    @staticmethod
    def _side_pressures(mach, p_inf, gamma, theta_front, theta_aft) -> tuple[tuple[float, float], list[str]]:
        notes = []
        m1, p1, flag1 = flow_deflect(mach, p_inf, theta_front, gamma)
        if flag1:
            notes.append(flag1)
        m2, p2, flag2 = flow_deflect(m1, p1, theta_aft - theta_front, gamma)
        if flag2:
            notes.append(flag2)
        return (p1, p2), notes

    def _newtonian_nose(self, mach, alpha, beta, q, cg_from_nose):
        notes = ["modified Newtonian cone, sharp, leeward Cp = 0, used only for M>=3"]
        gamma = self.gamma
        L = float(self.g["nose_length_m"])
        half = math.radians(float(self.g["nose_half_angle_deg"]))
        Rb = L * math.tan(half)
        Lcyl = float(self.g.get("cylinder_length_m", 0.0))
        cpmax = cp_max_modified_newtonian(mach, gamma)
        v_hat = self._vhat(alpha, beta)
        n_az = 16
        F = np.zeros(3)
        M = np.zeros(3)
        # Cone
        slant = L / max(math.cos(half), 0.2)
        A_lat = math.pi * Rb * slant
        dA = A_lat / n_az
        s = 0.5 * L  # mid-generator station from the tip
        r_local = s * math.tan(half)
        for i in range(n_az):
            psi = 2.0 * math.pi * (i + 0.5) / n_az
            # psi = 0 at +z (down). Outward normal has a forward component sin(half).
            e_r = np.array([0.0, math.sin(psi), math.cos(psi)])
            n_hat = math.sin(half) * np.array([1.0, 0.0, 0.0]) + math.cos(half) * e_r
            n_hat = n_hat / np.linalg.norm(n_hat)
            facing = float(np.dot(n_hat, v_hat))
            Cp = cpmax * facing * facing if facing > 0.0 else 0.0
            dF = -(q * Cp) * dA * n_hat  # gauge via Cp; leeward 0. Equivalent to (p-pinf).
            # Tip is cg_from_nose ahead of the CG on +x, i.e. x_tip = +cg_from_nose
            # only if the nose tip is the origin of cg_from_nose. Yes.
            x_tip = cg_from_nose
            r = np.array([x_tip - s * math.cos(half), r_local * math.sin(psi), r_local * math.cos(psi)])
            # Wait: s is along the axis aft, so body x decreases.
            r = np.array([x_tip - s, r_local * math.sin(psi), r_local * math.cos(psi)])
            F += dF
            M += np.cross(r, dF)
        # Cylinder: radial normals, same azimuth
        if Lcyl > 0.0 and Rb > 0.0:
            A_c = (2.0 * math.pi * Rb * Lcyl) / n_az
            x_start = cg_from_nose - L  # body x at cone base
            for i in range(n_az):
                psi = 2.0 * math.pi * (i + 0.5) / n_az
                n_hat = np.array([0.0, math.sin(psi), math.cos(psi)])
                facing = float(np.dot(n_hat, v_hat))
                Cp = cpmax * facing * facing if facing > 0.0 else 0.0
                dF = -(q * Cp) * A_c * n_hat
                r = np.array([x_start - 0.5 * Lcyl, Rb * math.sin(psi), Rb * math.cos(psi)])
                F += dF
                M += np.cross(r, dF)
        return F, M, notes

    def _fin(self, mach, beta, elevator_unused, q, cg_from_nose_m, atm_p):
        """Vertical fin. CL(beta) from the same section model. +beta -> force to -y.

        Force is applied at the section AC (c/4 subsonic, c/2 supersonic), which
        is the symmetric-section result from Anderson. Restoring yaw for +beta
        is +Mz (nose right) when the fin is behind the CG.
        """
        notes = []
        S = float(self.g["fin_area_m2"])
        span = float(self.g.get("fin_span_m", math.sqrt(S)))
        chord = float(self.g.get("fin_chord_m", S / max(span, 1e-6)))
        t_c = float(self.g.get("fin_thickness_ratio", self.g["thickness_ratio"]))
        # Use the wing coefficient path on a temporary one-off by calling wedge or subsonic
        # scalars. Build a small horizontal twin and read CL, CD via decompose.
        if mach >= 1.25:
            Fh, Mh, n = self._wedge_coeffs_only(mach, beta, 0.0, atm_p, q, S, chord, t_c)
            notes.extend(n)
            CL, CD = Fh
        else:
            m_use = min(mach, 0.85)
            AR = span * span / S
            # x_cg_from_le at the AC so this dummy moment is unused
            x_ac = 0.25 * chord
            Fw, Mw, n = self._subsonic_wing(m_use, beta, 0.0, q, S, chord, AR, x_ac)
            L, D, _Y = decompose_aero_force(Fw, beta, 0.0)
            CL = L / (q * S)
            CD = D / (q * S)
            notes.extend(n)
        L_fin = q * S * CL  # signed with beta; >0 for beta>0
        D_fin = q * S * abs(CD)
        v_hat = self._vhat(0.0, beta)
        # Side direction u_Y at alpha = 0 is +y when beta is small, but use the
        # same constructor as force_from_coefficients with a pure +Y test:
        u_Y = force_from_coefficients(0.0, 0.0, 1.0, 0.0, beta)
        # That returns a unit-ish Y direction only if we pass Y=1, D=0, L=0.
        if float(np.linalg.norm(u_Y)) < 1e-8:
            u_Y = np.array([0.0, 1.0, 0.0])
        else:
            u_Y = u_Y / np.linalg.norm(u_Y)
        F = -D_fin * v_hat - L_fin * u_Y
        x_ac_frac = 0.25 if mach < 1.0 else 0.5
        x_le = cg_from_nose_m - float(self.g["fin_x_from_nose_m"])
        x_ac = x_le - x_ac_frac * chord
        z_up = float(self.g.get("fin_z_up_m", 0.0))
        r = np.array([x_ac, 0.0, -z_up])
        M = np.cross(r, F)
        notes.append("fin: section CL(beta) placed at AC; no wing-fin interference")
        return F, M, notes

    def _wedge_coeffs_only(self, mach, alpha, elevator, p_inf, q, S, chord, t_c):
        F, M, notes = self._wedge_wing(mach, alpha, elevator, p_inf, q, S, chord, t_c, x_le_body=0.0)
        L, D, _ = decompose_aero_force(F, alpha, 0.0)
        return (L / (q * S), D / (q * S)), notes
