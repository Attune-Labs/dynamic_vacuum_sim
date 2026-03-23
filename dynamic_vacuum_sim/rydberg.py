"""Standard hydrogen spectrum using the reduced-mass Rydberg model.

Implements the familiar 1/n² energy ladder and transition frequencies
using CODATA-2018 constants with the electron-proton reduced mass μ.

Reference values: Tables I and II of
    White et al., Phys. Rev. Research 8, 013264 (2026).
"""

from __future__ import annotations

from . import constants as C

# ---------------------------------------------------------------------------
# Named spectral series  (lower level → series name)
# ---------------------------------------------------------------------------

SERIES: dict[str, int] = {
    "lyman": 1,
    "balmer": 2,
    "paschen": 3,
    "brackett": 4,
    "pfund": 5,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def level_energy(n: int) -> dict[str, float]:
    """Compute hydrogenic bound-state energy |E_n| for principal quantum number *n*.

    Uses the reduced-mass Rydberg formula (Table I of White et al.)::

        |E_n| = h c R_H / n²

    where R_H = R_∞ · (μ / m_e) is the reduced-mass Rydberg constant.

    Parameters
    ----------
    n : int  (≥ 1)
        Principal quantum number.

    Returns
    -------
    dict with keys:
        ``n``             – echo of input (dimensionless)
        ``energy_eV``     – |E_n|  [eV]
        ``energy_J``      – |E_n|  [J]
        ``frequency_Hz``  – f_n = |E_n| / h  [Hz]
        ``frequency_PHz`` – f_n  [PHz]

    Raises
    ------
    ValueError
        If *n* is not a positive integer.

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Table I, Eq. (20).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")

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
    }


def transition(n_upper: int, n_lower: int) -> dict[str, float]:
    """Compute frequency and vacuum wavelength for a hydrogenic transition.

    Eq. (20) of White et al.::

        f = c R_H (1/n_lower² − 1/n_upper²)

    Parameters
    ----------
    n_upper : int  (> n_lower)
        Upper principal quantum number.
    n_lower : int  (≥ 1)
        Lower principal quantum number.

    Returns
    -------
    dict with keys:
        ``n_upper``         – echo (dimensionless)
        ``n_lower``         – echo (dimensionless)
        ``frequency_Hz``    – transition frequency  [Hz]
        ``frequency_PHz``   – transition frequency  [PHz]
        ``wavelength_nm``   – vacuum wavelength  [nm]

    Raises
    ------
    TypeError
        If *n_upper* or *n_lower* is not an integer.
    ValueError
        If *n_lower* < 1 or *n_upper* ≤ *n_lower*.

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Eq. (20), Table II.
    """
    if not isinstance(n_upper, int) or not isinstance(n_lower, int):
        raise TypeError("n_upper and n_lower must be integers")
    if n_lower < 1:
        raise ValueError(f"n_lower must be ≥ 1, got {n_lower}")
    if n_upper <= n_lower:
        raise ValueError(f"n_upper must be > n_lower, got {n_upper} ≤ {n_lower}")

    freq_hz = C.C_LIGHT * C.R_H * (1.0 / n_lower**2 - 1.0 / n_upper**2)
    freq_phz = freq_hz / C.PHZ
    wavelength_m = C.C_LIGHT / freq_hz
    wavelength_nm = wavelength_m * 1e9

    return {
        "n_upper": n_upper,
        "n_lower": n_lower,
        "frequency_Hz": freq_hz,
        "frequency_PHz": freq_phz,
        "wavelength_nm": wavelength_nm,
    }


def series(name: str, n_max: int = 7) -> list[dict[str, float]]:
    """Return all transitions in a named spectral series up to *n_max*.

    Computes transition frequencies and wavelengths via :func:`transition`
    for n_upper = n_lower + 1 … n_max, where n_lower is the series-defining
    lower level (Lyman → 1, Balmer → 2, etc.).

    Parameters
    ----------
    name : str
        One of ``"lyman"``, ``"balmer"``, ``"paschen"``,
        ``"brackett"``, ``"pfund"`` (case-insensitive).
    n_max : int
        Highest upper level to include (must be > n_lower for the
        chosen series).

    Returns
    -------
    list of dicts – same keys as :func:`transition`.

    Raises
    ------
    ValueError
        If *name* is not a recognized series.

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Table II.
    """
    key = name.strip().lower()
    if key not in SERIES:
        raise ValueError(
            f"Unknown series {name!r}. Choose from {list(SERIES.keys())}"
        )
    n_lower = SERIES[key]
    return [transition(n2, n_lower) for n2 in range(n_lower + 1, n_max + 1)]
