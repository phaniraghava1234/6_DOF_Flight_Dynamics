# References

Equations in the code follow these sources. Page numbers are not invented.
Empirical constants that are quoted differently in secondary sources
(Sutton-Graves dimensional K, rolling friction, Oswald efficiency, fuel-air
limit) are user inputs.

1. NOAA, NASA, USAF. U.S. Standard Atmosphere, 1976.
   Geopotential height, layer lapse rates, hydrostatic equation with constant g0.
2. National Imagery and Mapping Agency. Department of Defense World Geodetic
   System 1984 (NIMA TR8350.2). mu, a, Earth rotation rate.
3. Vallado, D. A. Fundamentals of Astrodynamics and Applications, 4th ed.
   Microcosm, 2013. Two-body and J2 acceleration, orbital elements, ECEF/ECI.
4. Anderson, J. D. Modern Compressible Flow, 3rd ed. McGraw-Hill.
   Isentropic flow, normal and oblique shocks, Prandtl-Meyer, Rayleigh line,
   mass-flow parameter.
5. Anderson, J. D. Hypersonic and High-Temperature Gas Dynamics, 2nd ed. AIAA.
   Shock-expansion, modified Newtonian flow, Eckert reference temperature.
6. Anderson, J. D. Fundamentals of Aerodynamics. McGraw-Hill.
   Prandtl-Glauert, finite wing, Ackeret, aerodynamic center at c/4 subsonic
   and c/2 supersonic for a symmetric section.
7. Anderson, J. D. Aircraft Performance and Design. McGraw-Hill.
   Level-flight trim, Ps, ground roll, Breguet range, V-n ideas.
8. Glauert, H. The Elements of Aerofoil and Airscrew Theory. Cambridge, 1926.
   Thin-airfoil section. The numerical section here is a 1/4-3/4 vortex panel.
9. Katz, J. and Plotkin, A. Low-Speed Aerodynamics, 2nd ed. Cambridge.
   Vortex panel / 1/4-3/4 rule.
10. Eckert, E. R. G. "Engineering Relations for Friction and Heat Transfer to
    Surfaces in High Velocity Flow." Journal of the Aeronautical Sciences, 1955.
    Reference-temperature form as quoted by Anderson (Hypersonic...).
11. Schlichting, H. and Gersten, K. Boundary-Layer Theory. Springer.
    Turbulent flat-plate skin friction Cf = 0.455 / (log10 Re)^2.58.
    Laminar Blasius mean Cf = 1.328 / sqrt(Re).
12. Lees, L. "Hypersonic Flow." Fifth International Aeronautical Conference, 1955.
    Local-inclination / Newtonian windward pressure. Leeward Cp = 0.
13. Stevens, B. L., Lewis, F. L., and Johnson, E. N. Aircraft Control and
    Simulation, 3rd ed. Wiley, 2016.
    alpha, beta, body axes, wind-axis force resolution.
14. Zipfel, P. H. Modeling and Simulation of Aerospace Vehicle Dynamics. AIAA.
    Inertial 6DOF structure, quaternion attitude.
15. Diebel, J. "Representing Attitude: Euler Angles, Unit Quaternions, and
    Rotation Vectors." Stanford Technical Report, 2006.
    Scalar-first quaternion kinematics and DCM.
16. Schaub, H. and Junkins, J. L. Analytical Mechanics of Space Systems. AIAA.
    Gravity-gradient torque 3 mu / r^3 * r_hat x I r_hat.
17. Thomson, W. T. Introduction to Space Dynamics. Dover.
    Variable-mass rigid body: d(I w)/dt = M_ext, thrust accounted as an
    external force when exhaust momentum is in the thrust model.
18. Sutton, G. P. and Biblarz, O. Rocket Propulsion Elements. Wiley.
    F = mdot*ve + (pe - pa)*Ae, altitude thrust, Isp.
19. Mattingly, J. D. Elements of Propulsion: Gas Turbines and Rockets. AIAA.
    Ideal Brayton turbojet/ramjet, corrected mass flow, TSFC, overall efficiency.
20. Heiser, W. H. and Pratt, D. T. Hypersonic Airbreathing Propulsion. AIAA, 1994.
    Ramjet/scramjet station analysis and streamtube thrust
    F = mdot*((1+f)*Ve - V0) at perfect expansion.
21. Hill, P. G. and Peterson, C. R. Mechanics and Thermodynamics of Propulsion.
    Addison-Wesley. Steady-flow burner energy balance.
22. Sutton, K. and Graves, R. A. "A General Stagnation-Point Convective-Heating
    Equation for Arbitrary Gas Mixtures." NASA TR R-376, 1971.
    q_dot = K * sqrt(rho/Rn) * V^3. Dimensional K is a user input.
23. Rutowski, E. S. "Energy Approach to the General Aircraft Performance
    Problem." Journal of the Aeronautical Sciences, 1954.
    Ps = (T - D)V/W.
24. Bryson, A. E., Desai, M. N., and Hoffman, W. C. "Energy-State Approximation
    in Performance Optimization of Supersonic Aircraft." Journal of Aircraft, 1969.
25. Betts, J. T. "Survey of Numerical Methods for Trajectory Optimization."
    Journal of Guidance, Control, and Dynamics, 21(2), 1998.
    Direct shooting with parameterized controls.
26. Lawden, D. F. Optimal Trajectories for Space Navigation. Butterworths, 1963.
    Linear-tangent steering in vacuum. Implemented as an optional guidance law.
27. Vinh, N. X. Optimal Trajectories in Atmospheric Flight. Elsevier, 1981.
    Atmospheric point-mass flight-path problem class.
28. Breguet, L. range equation as given by Anderson, Aircraft Performance and
    Design: R = (V/c)(L/D) ln(W_i/W_f) with c the weight-specific fuel flow.

Not modeled, on purpose: MIL-E-5007 inlet recovery schedules, Sears-Haack wave
drag coefficient tables, base-pressure correlations, real-gas equilibrium air,
inlet unstart dynamics beyond a choked-mass-flow flag, aeroelasticity,
runway contact in 6DOF, and WGS-84 ellipsoid geodesy (spherical Earth of
radius a is stated and used).
