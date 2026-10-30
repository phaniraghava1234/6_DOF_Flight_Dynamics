# spaceplane_6dof

Modular preliminary-design flight mechanics for a supersonic / hypersonic
winged vehicle or Earth-to-orbit spaceplane.

Stack
- Earth: US Standard Atmosphere 1976 (lapse-rate model through 86 km), WGS-84
  point-mass + J2 gravity, rotating spherical Earth.
- Aero: subsonic vortex-panel section + Prandtl-Glauert + finite-wing factor;
  supersonic/hypersonic shock-expansion double wedge; modified-Newtonian nose
  for M >= 3; Eckert reference-temperature skin friction.
- Propulsion: ideal turbojet (corrected flow), ideal ramjet, constant-area
  Rayleigh scramjet, altitude-throttle rocket.
- Dynamics: inertial-frame translational EOM (so Coriolis is not added by
  hand) and body-frame Euler equations with quaternions. 3DOF ascent uses the
  same aero/propulsion with kinematic angle of attack.
- Performance: specific excess power, envelope, Breguet estimate, ground roll.
- Guidance: Rutowski energy climb on the airbreathing segment; gravity-turn
  pull-up then coast on the rocket segment; optional SLSQP tuning of the
  pull-up parameters (Betts direct shooting with a low-order control law).
- Safety: path constraints on dynamic pressure, Sutton-Graves stagnation
  heating, load factor, angle of attack, sideslip, and radiated wall
  temperature. Free 6DOF only monitors them. Mission mode can apply a simple
  envelope-protection governor (a control law, not a physical force).

What this is not
- Not POST, OTIS, or ASTOS.
- Not a CFD code and not a combustor design code.
- Calorically perfect gas, gamma = 1.4, is not valid in a real scramjet
  combustor (Anderson, Hypersonic and High-Temperature Gas Dynamics).
- Transonic coefficients are linearly blended between M = 0.85 and M = 1.25.
  That bridge is numerical continuity only. Transonic drag rise is not predicted.
- The file data/example_vehicle.json is an illustrative problem, not a
  published vehicle. Replace geometry, inertia, capture area, turbine
  temperatures, and cd_misc with test or CFD data before any design conclusion.
- Above 86 km the 1976 lapse-rate atmosphere stops. Density is set to zero
  rather than inventing a thermosphere. Aero and airbreathing thrust turn off.

Run
    python main.py --self-test
    python main.py --vehicle data/example_vehicle.json
    python main.py --vehicle data/example_vehicle.json --task mission
    python main.py --vehicle data/example_vehicle.json --task optimize
    python main.py --vehicle data/example_vehicle.json --task sim6dof
    python main.py --vehicle data/example_vehicle.json --task performance
    python main.py --vehicle data/example_vehicle.json --task aero --mach 2 --alpha-deg 4
    python main.py --vehicle data/example_vehicle.json --task propulsion --mach 6 --mode scramjet

Outputs go to outputs/*.csv.

Frames
- Inertial integration frame: epoch ECI. At t = 0 it coincides with ECEF.
  Earth rate is about +z. This is not a J2000 reduction.
- Body: x forward, y right, z down, right-handed.
  Positive pitch rate and My are nose-up.
- Alpha positive: nose above the air-relative velocity. Beta positive:
  air-relative velocity component to the right.

6DOF "free" means fixed controls and no stability augmentation. Attitude is
integrated. 3DOF mission attitude is kinematic.
