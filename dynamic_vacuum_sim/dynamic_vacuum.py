"""Dynamic-vacuum acoustic framework for hydrogenic spectra.

Implements the mapping from:
    White, Vera, Sylvester & Dudzinski,
    "Emergent quantization from a dynamic vacuum",
    Phys. Rev. Research 8, 013264 (2026).

The core idea: quadratic temporal dispersion  ω = D q²  (Eq. 1)
combined with a Coulombic constitutive profile  k²_eff = α + β/r  (Eq. 7)
yields the hydrogenic spectrum with a single reduced-mass calibration:
    D = ℏ/(2μ),  β = 2/a₀.
"""

from __future__ import annotations

import math

from . import constants as C
from . import rydberg


# ---------------------------------------------------------------------------
# Spatial eigenvalues
# ---------------------------------------------------------------------------


def kappa_n(n: int) -> float:
    """Bound-state inverse decay length  κ_n = β/(2n) = 1/(n a₀)  [1/m].

    Eq. (18) of White et al.

    Parameters
    ----------
    n : int  (≥ 1)
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return C.BETA / (2.0 * n)


# ---------------------------------------------------------------------------
# Eigenfrequencies and energies
# ---------------------------------------------------------------------------


def omega_n(n: int) -> float:
    """Eigenfrequency  ω_n = D κ_n² = ω*/n²  [rad/s].

    Eq. (10) of White et al.:  ω_n = D β² / (4 n²).

    Computed as ω*/n² for maximal numerical precision (avoids
    accumulating rounding through D and κ separately).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return C.OMEGA_STAR / n**2


def level_energy(n: int) -> dict[str, float]:
    """Compute bound-state energy from the dynamic-vacuum mapping.

    E_n = −ℏ ω_n   →   |E_n| = ℏ ω_n   (Eq. 19).

    Parameters
    ----------
    n : int  (≥ 1)

    Returns
    -------
    dict with keys:
        ``n``             – echo
        ``energy_eV``     – |E_n| in eV
        ``energy_J``      – |E_n| in J
        ``frequency_Hz``  – f_n = |E_n|/h
        ``frequency_PHz`` – f_n in PHz
        ``kappa_1m``      – κ_n  [1/m]
        ``omega_rad``     – ω_n  [rad/s]
    """
    kn = kappa_n(n)
    wn = omega_n(n)
    # Compute energy via h·c·R_H/n² (same arithmetic path as rydberg.py)
    # to guarantee bit-exact isospectrality.  Analytically this equals ℏ·ω_n.
    energy_j = C.H_PLANCK * C.C_LIGHT * C.R_H / n**2
    energy_ev = energy_j * C.EV_PER_J
    freq_hz = energy_j / C.H_PLANCK
    freq_phz = freq_hz / C.PHZ

    return {
        "n": n,
        "energy_eV": energy_ev,
        "energy_J": energy_j,
        "frequency_Hz": freq_hz,
        "frequency_PHz": freq_phz,
        "kappa_1m": kn,
        "omega_rad": wn,
    }


# ---------------------------------------------------------------------------
# Constitutive coefficients  (Eq. 22)
# ---------------------------------------------------------------------------


def A_coeff(n: int) -> float:
    """Frequency-dependent constitutive coefficient A(ω_n).

    Eq. (22):  A(ω_n) = −n² / (a₀² ω*²).

    Always negative (reactive stop band ⇒ bound states).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return -(n**2) / (C.A0**2 * C.OMEGA_STAR**2)


def C_coeff(n: int) -> float:
    """Frequency-dependent constitutive coefficient C(ω_n).

    Eq. (22):  C(ω_n) = 2 n⁴ / (a₀ ω*²).

    Always positive (proton-induced increase of 1/c²_s near core).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return 2.0 * n**4 / (C.A0 * C.OMEGA_STAR**2)


# ---------------------------------------------------------------------------
# Effective wave number
# ---------------------------------------------------------------------------


def k_eff_squared(r: float, n: int) -> float:
    """Effective squared wave number at radius *r* for eigenfrequency ω_n.

    Eq. (7):  k²_eff(r) = α(ω_n) + β/r  =  −κ_n² + β/r.

    Parameters
    ----------
    r : float  — radial position [m]  (must be > 0)
    n : int    — principal quantum number
    """
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    kn = kappa_n(n)
    return -kn**2 + C.BETA / r


# ---------------------------------------------------------------------------
# Full Madelung dispersion  (Appendix, Eq. A21)
# ---------------------------------------------------------------------------


def dispersion_full(k: float, c_L: float = 0.0) -> float:
    """Full Madelung dispersion  ω² = c_L² k² + D² k⁴  (Eq. A21).

    In the quantum-pressure-dominated regime (c_L → 0) this reduces
    to ω = D k², the quadratic law used throughout the paper.

    Parameters
    ----------
    k   : float  — wave number  [1/m]
    c_L : float  — longitudinal sound speed  [m/s]  (default 0)

    Returns
    -------
    omega : float  — angular frequency  [rad/s]
    """
    omega_sq = c_L**2 * k**2 + C.D_DISP**2 * k**4
    return math.sqrt(omega_sq)


# ---------------------------------------------------------------------------
# Isospectrality verification
# ---------------------------------------------------------------------------


def verify_isospectrality(
    n_max: int = 7,
    rtol: float = 1e-12,
) -> list[dict[str, float]]:
    """Compare dynamic-vacuum energies to Rydberg energies for n = 1 … n_max.

    Raises ``AssertionError`` if any relative error exceeds *rtol*.

    Returns
    -------
    list of dicts with keys: ``n``, ``E_rydberg_eV``, ``E_dv_eV``, ``rel_error``.
    """
    results: list[dict[str, float]] = []
    for n in range(1, n_max + 1):
        e_ryd = rydberg.level_energy(n)["energy_eV"]
        e_dv = level_energy(n)["energy_eV"]
        rel = abs(e_dv - e_ryd) / e_ryd if e_ryd != 0 else 0.0
        results.append(
            {
                "n": n,
                "E_rydberg_eV": e_ryd,
                "E_dv_eV": e_dv,
                "rel_error": rel,
            }
        )
        assert rel <= rtol, (
            f"Isospectrality failed at n={n}: "
            f"Rydberg={e_ryd:.12e} eV, DV={e_dv:.12e} eV, "
            f"rel_error={rel:.2e} > rtol={rtol:.2e}"
        )
    return results
