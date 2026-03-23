"""Tests for dynamic_vacuum.py — isospectrality and constitutive coefficients."""

import math
import pytest

from dynamic_vacuum_sim import dynamic_vacuum as dv
from dynamic_vacuum_sim import constants as C


# ------------------------------------------------------------------
# Isospectrality  (the central claim of the paper)
# ------------------------------------------------------------------

def test_isospectrality_n1_to_7():
    """Dynamic-vacuum energies must match Rydberg to < 1e-12 relative."""
    results = dv.verify_isospectrality(n_max=7, rtol=1e-12)
    for r in results:
        assert r["rel_error"] < 1e-12, f"n={r['n']}: rel_error={r['rel_error']}"


def test_isospectrality_n1_to_20():
    """Extend check to n=20."""
    results = dv.verify_isospectrality(n_max=20, rtol=1e-11)
    assert len(results) == 20


# ------------------------------------------------------------------
# kappa_n
# ------------------------------------------------------------------

def test_kappa_n_equals_inv_n_a0():
    for n in range(1, 8):
        expected = 1.0 / (n * C.A0)
        assert abs(dv.kappa_n(n) - expected) / expected < 1e-14


def test_kappa_n_equals_beta_over_2n():
    for n in range(1, 8):
        expected = C.BETA / (2.0 * n)
        assert abs(dv.kappa_n(n) - expected) / expected < 1e-14


# ------------------------------------------------------------------
# omega_n
# ------------------------------------------------------------------

def test_omega_n_scales_as_inv_n_squared():
    w1 = dv.omega_n(1)
    for n in range(2, 8):
        expected = w1 / n**2
        assert abs(dv.omega_n(n) - expected) / expected < 1e-13


def test_omega_1_equals_omega_star():
    assert abs(dv.omega_n(1) - C.OMEGA_STAR) / C.OMEGA_STAR < 1e-14


# ------------------------------------------------------------------
# Constitutive coefficients  (Eq. 22)
# ------------------------------------------------------------------

def test_A_coeff_negative():
    """A(ω_n) < 0 for all n (reactive stop band)."""
    for n in range(1, 8):
        assert dv.A_coeff(n) < 0, f"A(ω_{n}) should be negative"


def test_C_coeff_positive():
    """C(ω_n) > 0 for all n."""
    for n in range(1, 8):
        assert dv.C_coeff(n) > 0, f"C(ω_{n}) should be positive"


def test_A_coeff_scales_n_squared():
    """| A(ω_n) | ∝ n²."""
    A1 = abs(dv.A_coeff(1))
    for n in range(2, 8):
        ratio = abs(dv.A_coeff(n)) / A1
        assert abs(ratio - n**2) / n**2 < 1e-12


def test_C_coeff_scales_n_fourth():
    """C(ω_n) ∝ n⁴."""
    C1 = dv.C_coeff(1)
    for n in range(2, 8):
        ratio = dv.C_coeff(n) / C1
        assert abs(ratio - n**4) / n**4 < 1e-12


# ------------------------------------------------------------------
# k_eff_squared
# ------------------------------------------------------------------

def test_k_eff_squared_at_large_r():
    """At very large r, k²_eff → −κ_n² (evanescent)."""
    r_large = 1e6 * C.A0
    for n in range(1, 5):
        kn = dv.kappa_n(n)
        k2 = dv.k_eff_squared(r_large, n)
        assert abs(k2 - (-kn**2)) / kn**2 < 1e-4


# ------------------------------------------------------------------
# Full dispersion  (Eq. A21)
# ------------------------------------------------------------------

def test_dispersion_quadratic_limit():
    """With c_L=0, ω = D k²."""
    k = 1e10
    omega = dv.dispersion_full(k, c_L=0.0)
    expected = C.D_DISP * k**2
    assert abs(omega - expected) / expected < 1e-14


def test_dispersion_acoustic_limit():
    """With large c_L, low k: ω ≈ c_L k."""
    c_L = 1e3
    k = 1.0
    omega = dv.dispersion_full(k, c_L=c_L)
    expected = c_L * k
    assert abs(omega - expected) / expected < 1e-6


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_kappa_invalid():
    with pytest.raises(ValueError):
        dv.kappa_n(0)
