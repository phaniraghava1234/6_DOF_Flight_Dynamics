"""US Standard Atmosphere 1976, geopotential lapse model through 86 km.

Hydrostatic equation with constant g0, as specified by US76:
    dT/dH = L
    dp/p = -g0 dH / (R T)
Closed form:
    L != 0: p = pb * (T/Tb) ** (-g0 / (L R))
    L == 0: p = pb * exp(-g0 (H-Hb) / (R Tb))
Geometric altitude h and geopotential H:
    H = r0 * h / (r0 + h)
The kinetic-theory extension above 86 km is not implemented. Density is 0
there so a false thermosphere is not invented.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .constants import (
    GAMMA_AIR,
    G0,
    P0_STD,
    R0_GEOPOTENTIAL,
    R_AIR,
    SUTHERLAND_B,
    SUTHERLAND_S,
    T0_STD,
)

# (H_base_m, lapse_K_per_m). Temperature at the next base follows from L.
# Values are the US76 layer breakpoints through H = 84852 m (~86 km geometric).
_LAYERS = (
    (0.0, -0.0065),
    (11000.0, 0.0),
    (20000.0, 0.0010),
    (32000.0, 0.0028),
    (47000.0, 0.0),
    (51000.0, -0.0028),
    (71000.0, -0.0020),
    (84852.0, 0.0),
)


@dataclass(frozen=True)
class AtmosphereState:
    h_m: float
    T_K: float
    p_Pa: float
    rho: float
    a_mps: float
    mu: float
    geopotential_m: float
    in_model: bool


def geopotential_height(h_geometric_m: float) -> float:
    h = max(float(h_geometric_m), -1000.0)
    return R0_GEOPOTENTIAL * h / (R0_GEOPOTENTIAL + h)


def _layer_table() -> list[tuple[float, float, float, float]]:
    """Return (H_base, T_base, lapse, p_base) integrated from sea level."""
    rows = []
    H_b, T_b, p_b = 0.0, T0_STD, P0_STD
    for i, (H_i, L_i) in enumerate(_LAYERS):
        if i == 0:
            rows.append((H_i, T0_STD, L_i, P0_STD))
            continue
        H_prev, L_prev = _LAYERS[i - 1]
        T_b = rows[-1][1] + L_prev * (H_i - H_prev)
        p_prev = rows[-1][3]
        T_prev = rows[-1][1]
        if abs(L_prev) < 1e-12:
            p_b = p_prev * math.exp(-G0 * (H_i - H_prev) / (R_AIR * T_prev))
        else:
            p_b = p_prev * (T_b / T_prev) ** (-G0 / (L_prev * R_AIR))
        rows.append((H_i, T_b, L_i, p_b))
    return rows


_TABLE = _layer_table()


def sutherland_mu(T_K: float) -> float:
    T = max(T_K, 1.0)
    return SUTHERLAND_B * T ** 1.5 / (T + SUTHERLAND_S)


def us76(h_geometric_m: float) -> AtmosphereState:
    h = float(h_geometric_m)
    H = geopotential_height(max(h, 0.0))
    if h > 86000.0:
        top = _TABLE[-1]
        T = top[1]
        return AtmosphereState(h, T, 0.0, 0.0, math.sqrt(GAMMA_AIR * R_AIR * T), sutherland_mu(T), H, False)
    # Find layer
    idx = 0
    for i in range(len(_TABLE) - 1):
        if H >= _TABLE[i][0]:
            idx = i
    H_b, T_b, L, p_b = _TABLE[idx]
    if idx < len(_TABLE) - 1 and H > _TABLE[-1][0]:
        H = _TABLE[-1][0]
    T = T_b + L * (H - H_b)
    T = max(T, 1.0)
    if abs(L) < 1e-12:
        p = p_b * math.exp(-G0 * (H - H_b) / (R_AIR * T_b))
    else:
        p = p_b * (T / T_b) ** (-G0 / (L * R_AIR))
    rho = p / (R_AIR * T)
    a = math.sqrt(GAMMA_AIR * R_AIR * T)
    return AtmosphereState(h, T, p, rho, a, sutherland_mu(T), H, True)
