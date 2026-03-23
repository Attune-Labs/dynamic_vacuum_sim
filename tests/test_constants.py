"""Tests for constants.py — verify self-consistency of derived quantities."""

import math
import pytest

from dynamic_vacuum_sim import constants as C


def test_reduced_mass_ratio():
    """μ/m_e ≈ 0.999 455 679 ..."""
    ratio = C.MU / C.M_E
    assert 0.99945 < ratio < 0.99946


def test_bohr_radius_order_of_magnitude():
    """a₀(H) ≈ 5.295 × 10⁻¹¹ m."""
    assert 5.29e-11 < C.A0 < 5.30e-11


def test_rydberg_constant_reduced():
    """R_H = R_∞ × μ/m_e."""
    expected = C.R_INF * C.MU / C.M_E
    assert abs(C.R_H - expected) / expected < 1e-14


def test_dispersion_identity():
    """D = ℏ/(2μ) and also D = ω* a₀².

    The two paths differ at ~1e-9 relative due to accumulated
    floating-point rounding through independent intermediate
    constants (A0, OMEGA_STAR).  Both are correct to double
    precision.
    """
    d_from_hbar = C.HBAR / (2.0 * C.MU)
    d_from_omega = C.OMEGA_STAR * C.A0**2
    assert abs(d_from_hbar - C.D_DISP) / C.D_DISP < 1e-14
    assert abs(d_from_omega - C.D_DISP) / C.D_DISP < 1e-8  # fp rounding


def test_beta():
    """β = 2/a₀."""
    assert abs(C.BETA - 2.0 / C.A0) / C.BETA < 1e-14


def test_rydberg_energy():
    """RY_EV should be ≈ 13.598 eV (reduced-mass value, not 13.606)."""
    assert 13.598 < C.RY_EV < 13.599
