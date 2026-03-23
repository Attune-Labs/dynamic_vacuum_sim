"""Hydrogenic radial eigenfunctions R_{nℓ}(r).

Closed-form evaluation via associated Laguerre polynomials,
following Eq. (13a) and the normalization in Sec. III of
    White et al., Phys. Rev. Research 8, 013264 (2026):

    R_{nℓ}(r) = N_{nℓ} (2κ_n r)^ℓ  exp(−κ_n r)  L^{2ℓ+1}_{n−ℓ−1}(2κ_n r)

    N_{nℓ} = (2κ_n)^{3/2} √[(n−ℓ−1)! / (2n (n+ℓ)!)]

Uses ``scipy.special.genlaguerre`` for the generalized Laguerre polynomial.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import special

from . import constants as C


def _kappa(n: int, model: str) -> float:
    """Return κ_n for the chosen model (both give the same value)."""
    # By exact isospectrality, κ_n = 1/(n a₀) regardless of model.
    # We keep the model parameter for API symmetry and future extensions.
    if model not in ("rydberg", "dynamic_vacuum"):
        raise ValueError(f"model must be 'rydberg' or 'dynamic_vacuum', got {model!r}")
    return 1.0 / (n * C.A0)


def _validate_quantum_numbers(n: int, ell: int) -> None:
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(ell, int) or ell < 0:
        raise ValueError(f"ell must be a non-negative integer, got {ell!r}")
    if ell >= n:
        raise ValueError(f"ell must be < n, got ell={ell} with n={n}")


def R_nl(
    n: int,
    ell: int,
    r: NDArray[np.float64],
    model: str = "rydberg",
) -> NDArray[np.float64]:
    """Normalized hydrogenic radial wave function R_{nℓ}(r)  [1/m^{3/2}].

    Eq. (13a) of White et al.::

        R_{nℓ}(r) = N_{nℓ} (2κr)^ℓ  exp(−κr)  L^{2ℓ+1}_{n−ℓ−1}(2κr)

    Normalization (Sec. III, below Eq. 17)::

        N_{nℓ} = (2κ)^{3/2}  √[(n−ℓ−1)! / (2n (n+ℓ)!)]

    satisfying  ∫₀^∞ |R_{nℓ}|² r² dr = 1.

    Parameters
    ----------
    n : int
        Principal quantum number (≥ 1).
    ell : int
        Orbital angular momentum quantum number (0 ≤ ℓ < n).
    r : ndarray
        Radial positions  [m].
    model : str
        ``"rydberg"`` or ``"dynamic_vacuum"`` (identical by isospectrality;
        included for API symmetry and future extensions).

    Returns
    -------
    R : ndarray
        Radial wave function values  [1/m^{3/2}], same shape as *r*.

    Raises
    ------
    ValueError
        If quantum numbers violate n ≥ 1, ℓ ≥ 0, or ℓ < n.

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Eq. (13a), Sec. III.
    """
    _validate_quantum_numbers(n, ell)
    r = np.asarray(r, dtype=np.float64)
    kn = _kappa(n, model)

    # Degree of associated Laguerre polynomial
    nr = n - ell - 1  # radial quantum number (number of nodes)

    # Normalization  (Sec. III, below Eq. 17)
    norm = (2.0 * kn) ** 1.5 * math.sqrt(
        math.factorial(nr) / (2.0 * n * math.factorial(n + ell))
    )

    rho = 2.0 * kn * r  # dimensionless variable

    # Generalized Laguerre polynomial  L^{2ℓ+1}_{n−ℓ−1}(ρ)
    lag = special.genlaguerre(nr, 2 * ell + 1)

    return norm * rho**ell * np.exp(-rho / 2.0) * lag(rho)


def radial_probability_density(
    n: int,
    ell: int,
    r: NDArray[np.float64],
    model: str = "rydberg",
) -> NDArray[np.float64]:
    """Radial probability density r² |R_{nℓ}(r)|²  [1/m].

    The quantity whose integral over r from 0 to ∞ equals 1.
    Related to the acoustic pressure-amplitude envelope in the
    dynamic-vacuum picture (Sec. III of White et al.).

    Parameters
    ----------
    n, ell, r, model : same as :func:`R_nl`.

    Returns
    -------
    density : ndarray
        Probability density  [1/m], same shape as *r*.

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Sec. III.
    """
    R = R_nl(n, ell, r, model=model)
    return r**2 * R**2


def verify_orthonormality(
    n_max: int = 4,
    r_max_bohr: float = 200.0,
    n_pts: int = 8000,
) -> list[dict]:
    """Verify ⟨R_{n'ℓ} | R_{nℓ}⟩ = δ_{n'n} on a numerical radial grid.

    For each angular momentum ℓ = 0 … n_max − 1, compute the overlap
    integral for all pairs (n', n) with n', n ≥ ℓ + 1 using the
    trapezoidal rule.  Normalization convention follows Sec. III of
    White et al.

    Parameters
    ----------
    n_max : int
        Largest principal quantum number to include (default 4).
    r_max_bohr : float
        Upper integration limit in units of the Bohr radius a₀
        (dimensionless; default 200.0).
    n_pts : int
        Number of radial grid points (default 8000).

    Returns
    -------
    list of dicts with keys:
        ``n1``, ``n2``  – principal quantum numbers (dimensionless)
        ``ell``         – angular momentum quantum number (dimensionless)
        ``overlap``     – numerical overlap integral (dimensionless)
        ``passed``      – bool, True if |overlap − δ_{n1,n2}| < 1e-4

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Sec. III, Eq. (17).
    """
    r = np.linspace(0, r_max_bohr * C.A0, n_pts)
    dr = r[1] - r[0]
    results: list[dict] = []

    for ell in range(n_max):
        ns = list(range(ell + 1, n_max + 1))
        for i, n1 in enumerate(ns):
            R1 = R_nl(n1, ell, r)
            for n2 in ns[i:]:
                R2 = R_nl(n2, ell, r)
                # ∫ R_{n1,ℓ}(r) R_{n2,ℓ}(r) r² dr  (trapezoidal)
                integrand = R1 * R2 * r**2
                overlap = float(np.trapezoid(integrand, dx=dr))
                expected = 1.0 if n1 == n2 else 0.0
                passed = abs(overlap - expected) < 1e-4
                results.append(
                    {
                        "n1": n1,
                        "n2": n2,
                        "ell": ell,
                        "overlap": overlap,
                        "passed": passed,
                    }
                )
    return results
