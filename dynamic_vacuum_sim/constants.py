"""Physical constants (CODATA-2018) and derived reduced-mass Rydberg quantities.

All fundamental values from:
    Tiesinga et al., "CODATA recommended values of the fundamental physical
    constants: 2018", Rev. Mod. Phys. 93, 025010 (2021).

Derived quantities use the electron-proton reduced mass
    μ = m_e · m_p / (m_e + m_p).

The calibration follows White et al., Phys. Rev. Research 8, 013264 (2026),
Eqs. (11), (18), (21), (22).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Fundamental constants  (CODATA-2018, exact or best values)
# ---------------------------------------------------------------------------

HBAR: float = 6.626_070_15e-34 / (2.0 * math.pi)
"""Reduced Planck constant ℏ = h/(2π)  [J·s].

Derived from the exact CODATA-2018 value of h to maintain
arithmetic consistency: h ≡ 2π ℏ  exactly in this module.
"""

H_PLANCK: float = 6.626_070_15e-34
"""Planck constant h  [J·s]."""

M_E: float = 9.109_383_7015e-31
"""Electron mass m_e  [kg]."""

M_P: float = 1.672_621_923_69e-27
"""Proton mass m_p  [kg]."""

E_CHARGE: float = 1.602_176_634e-19
"""Elementary charge e  [C]."""

EPSILON_0: float = 8.854_187_8128e-12
"""Vacuum permittivity ε₀  [F/m]."""

C_LIGHT: float = 299_792_458.0
"""Speed of light in vacuum c  [m/s]."""

EV_PER_J: float = 1.0 / E_CHARGE
"""Conversion factor: electronvolts per joule."""

# ---------------------------------------------------------------------------
# Infinite-mass Rydberg constant  (CODATA-2018)
# ---------------------------------------------------------------------------

R_INF: float = 10_973_731.568_160
"""Rydberg constant R_∞  [1/m]."""

# ---------------------------------------------------------------------------
# Derived reduced-mass quantities
# ---------------------------------------------------------------------------

MU: float = M_E * M_P / (M_E + M_P)
"""Electron-proton reduced mass μ  [kg]."""

A0: float = 4.0 * math.pi * EPSILON_0 * HBAR**2 / (MU * E_CHARGE**2)
"""Bohr radius a₀(H) = 4πε₀ℏ² / (μe²)  [m].

Uses the reduced mass μ, not the bare electron mass.
Eq. following (21) of White et al.
"""

R_H: float = R_INF * MU / M_E
"""Reduced-mass Rydberg constant R_H = R_∞ · (μ/m_e)  [1/m]."""

OMEGA_STAR: float = 2.0 * math.pi * C_LIGHT * R_H
"""Reduced-mass Rydberg angular frequency ω* = 2π c R_H  [rad/s].

Calibration target: Eq. (10)/(11) of White et al.
"""

D_DISP: float = HBAR / (2.0 * MU)
"""Dispersion constant D = ℏ/(2μ)  [m²/s].

Eq. (1)/(11) of White et al.  Satisfies D = ω* a₀².
"""

BETA: float = 2.0 / A0
"""Coupling inverse length β = 2/a₀  [1/m].

Eq. (18) of White et al.
"""

RY_EV: float = H_PLANCK * C_LIGHT * R_H * EV_PER_J
"""Reduced-mass Rydberg energy h c R_H  [eV].

|E₁| for the hydrogen ground state.
"""

# ---------------------------------------------------------------------------
# Convenience: 1 PHz in Hz
# ---------------------------------------------------------------------------

PHZ: float = 1.0e15
"""One petahertz in hertz."""
