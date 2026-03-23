"""Matplotlib helpers for visualizing hydrogenic spectra and radial functions.

All plots use a clean, publication-ready style.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from . import constants as C
from . import rydberg
from . import dynamic_vacuum as dv
from . import radial

# Use non-interactive backend when no display is available
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Colour palette for spectral series
# ---------------------------------------------------------------------------

SERIES_COLORS: dict[str, str] = {
    "lyman": "#7B2FBE",
    "balmer": "#2196F3",
    "paschen": "#E53935",
    "brackett": "#FF9800",
    "pfund": "#4CAF50",
}


def plot_levels(
    n_max: int = 7,
    model: str = "rydberg",
    ax: Optional[plt.Axes] = None,
) -> matplotlib.figure.Figure:
    """Ladder diagram of hydrogenic energy levels E_1 … E_{n_max}.

    Draws horizontal bars at −|E_n| [eV] for each level, using either
    the standard Rydberg formula or the dynamic-vacuum mapping.

    Parameters
    ----------
    n_max : int
        Highest principal quantum number to show (default 7).
    model : str
        ``"rydberg"`` or ``"dynamic_vacuum"`` (default ``"rydberg"``).
    ax : matplotlib.axes.Axes or None
        Optional pre-existing axes; a new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 7))
    else:
        fig = ax.get_figure()

    energies: list[float] = []
    for n in range(1, n_max + 1):
        if model == "dynamic_vacuum":
            e = dv.level_energy(n)["energy_eV"]
        else:
            e = rydberg.level_energy(n)["energy_eV"]
        energies.append(-e)  # negative for bound states

    for n, e in enumerate(energies, start=1):
        ax.hlines(e, 0.2, 0.8, colors="k", linewidth=1.5)
        ax.text(0.85, e, f"n = {n}", va="center", fontsize=9)

    ax.set_xlim(0, 1.2)
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Hydrogen Energy Levels ({model})")
    ax.get_xaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_series(
    series_names: Optional[list[str]] = None,
    n_max: int = 7,
    ax: Optional[plt.Axes] = None,
) -> matplotlib.figure.Figure:
    """Grotrian-style diagram of selected spectral-line series.

    Draws vertical arrows between energy levels, colour-coded by
    series (Lyman, Balmer, Paschen, etc.).  Transition wavelengths
    follow Eq. (20) of White et al.

    Parameters
    ----------
    series_names : list of str or None
        Series to display (default ``["lyman", "balmer", "paschen"]``).
    n_max : int
        Highest upper level to include (default 7).
    ax : matplotlib.axes.Axes or None
        Optional pre-existing axes; a new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Eq. (20), Table II.
    """
    if series_names is None:
        series_names = ["lyman", "balmer", "paschen"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    # Pre-compute levels
    levels = {}
    for n in range(1, n_max + 1):
        levels[n] = -rydberg.level_energy(n)["energy_eV"]

    # Draw level bars
    for n, e in levels.items():
        ax.hlines(e, 0, len(series_names) + 1, colors="0.8", linewidth=0.8)
        ax.text(len(series_names) + 1.1, e, f"n={n}", va="center", fontsize=8)

    # Draw transitions
    for i, sname in enumerate(series_names):
        n_lower = rydberg.SERIES[sname.lower()]
        color = SERIES_COLORS.get(sname.lower(), "#333")
        x_pos = i + 0.8
        for n_upper in range(n_lower + 1, n_max + 1):
            e_low = levels[n_lower]
            e_high = levels[n_upper]
            ax.annotate(
                "",
                xy=(x_pos, e_low),
                xytext=(x_pos, e_high),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
            )
        # Label the series
        ax.text(
            x_pos,
            levels[1] - 1.5,
            sname.capitalize(),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(-0.2, len(series_names) + 2)
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Hydrogen Spectral Series")
    ax.get_xaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_radial(
    n: int,
    ell: int,
    model: str = "rydberg",
    r_max_bohr: Optional[float] = None,
    n_pts: int = 1000,
    ax: Optional[plt.Axes] = None,
) -> matplotlib.figure.Figure:
    """Plot radial probability density r² |R_{nℓ}|² vs r / a₀.

    Parameters
    ----------
    n : int
        Principal quantum number (≥ 1).
    ell : int
        Orbital angular momentum quantum number (0 ≤ ℓ < n).
    model : str
        ``"rydberg"`` or ``"dynamic_vacuum"`` (default ``"rydberg"``).
    r_max_bohr : float or None
        Upper r-axis limit in units of a₀ (auto-scaled to
        2.5 n² + 10 if None).
    n_pts : int
        Number of radial grid points (default 1000).
    ax : matplotlib.axes.Axes or None
        Optional pre-existing axes; a new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Eq. (13a), Sec. III.
    """
    if r_max_bohr is None:
        r_max_bohr = float(2.5 * n**2 + 10)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    r_m = np.linspace(1e-15, r_max_bohr * C.A0, n_pts)
    r_bohr = r_m / C.A0

    density = radial.radial_probability_density(n, ell, r_m, model=model)
    # Convert density from 1/m to 1/a₀ for display
    density_bohr = density * C.A0

    ax.plot(r_bohr, density_bohr, linewidth=1.5)
    ax.fill_between(r_bohr, density_bohr, alpha=0.15)
    ax.set_xlabel(r"$r\;/\;a_0$")
    ax.set_ylabel(r"$r^2 |R_{n\ell}(r)|^2$  (per $a_0$)")
    ax.set_title(f"Radial density  n={n}, ℓ={ell}  ({model})")
    ax.set_xlim(0, r_max_bohr)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_radial_comparison(
    n: int,
    ell: int,
    r_max_bohr: Optional[float] = None,
    n_pts: int = 1000,
    ax: Optional[plt.Axes] = None,
) -> matplotlib.figure.Figure:
    """Overlay Rydberg and dynamic-vacuum radial probability densities.

    A visual confirmation of exact isospectrality: both models produce
    identical r² |R_{nℓ}(r)|² curves by construction (Sec. IV of
    White et al.).

    Parameters
    ----------
    n : int
        Principal quantum number (≥ 1).
    ell : int
        Orbital angular momentum quantum number (0 ≤ ℓ < n).
    r_max_bohr : float or None
        Upper r-axis limit in units of a₀ (auto-scaled to
        2.5 n² + 10 if None).
    n_pts : int
        Number of radial grid points (default 1000).
    ax : matplotlib.axes.Axes or None
        Optional pre-existing axes; a new figure is created if None.

    Returns
    -------
    fig : matplotlib.figure.Figure

    References
    ----------
    White et al., Phys. Rev. Research 8, 013264 (2026), Sec. IV.
    """
    if r_max_bohr is None:
        r_max_bohr = float(2.5 * n**2 + 10)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    r_m = np.linspace(1e-15, r_max_bohr * C.A0, n_pts)
    r_bohr = r_m / C.A0

    for model, ls, lbl in [
        ("rydberg", "-", "Rydberg"),
        ("dynamic_vacuum", "--", "Dynamic vacuum"),
    ]:
        density = radial.radial_probability_density(n, ell, r_m, model=model)
        density_bohr = density * C.A0
        ax.plot(r_bohr, density_bohr, ls, linewidth=1.5, label=lbl)

    ax.set_xlabel(r"$r\;/\;a_0$")
    ax.set_ylabel(r"$r^2 |R_{n\ell}(r)|^2$  (per $a_0$)")
    ax.set_title(f"Isospectrality check  n={n}, ℓ={ell}")
    ax.legend()
    ax.set_xlim(0, r_max_bohr)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig
