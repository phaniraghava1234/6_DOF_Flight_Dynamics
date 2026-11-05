"""Vehicle loader.

The JSON is the vehicle. Nothing aerodynamic or propulsive is hard-coded
beyond the models themselves. See data/example_vehicle.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from .aerodynamics import Aerodynamics
from .propulsion import Propulsion


class Vehicle:
    def __init__(self, cfg: dict, source: str = ""):
        self.cfg = cfg
        self.source = source
        self._validate()
        self.aero = Aerodynamics(cfg["geometry"])
        self.prop = Propulsion(cfg["propulsion"])

    def _validate(self):
        g = self.cfg["geometry"]
        if abs(g["wing_span_m"] * g["wing_chord_m"] - g["wing_area_m2"]) / g["wing_area_m2"] > 0.05:
            raise ValueError("wing_span * wing_chord must match wing_area within 5% (rectangular wing assumption)")
        m = self.cfg["mass"]
        for k in ("dry_kg", "fuel_airbreathing_kg", "rocket_propellant_kg"):
            if float(m[k]) < 0:
                raise ValueError(f"mass.{k} must be >= 0")
        if float(self.cfg["propulsion"]["rocket_thrust_vac_n"]) <= float(self.cfg["propulsion"]["rocket_thrust_sl_n"]):
            raise ValueError("vacuum thrust must exceed sea-level thrust")
        cg = self.cfg["cg_from_nose_m"]
        chord = float(g["wing_chord_m"])
        le = float(g["wing_le_from_nose_m"])
        for label, xcg in cg.items():
            h = (float(xcg) - le) / chord
            if h <= 0.0 or h >= 0.6:
                # Not fatal: a user may be exploring an unstable cg. Warn by exception message only if absurd.
                if h < -0.5 or h > 1.5:
                    raise ValueError(f"cg {label} is not near the wing; check cg_from_nose_m")

    @staticmethod
    def from_json(path: str | Path) -> "Vehicle":
        p = Path(path)
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return Vehicle(cfg, source=str(p))
