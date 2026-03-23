"""Tests for radial.py — wave functions, nodes, and orthonormality."""

import numpy as np
import pytest

from dynamic_vacuum_sim import radial
from dynamic_vacuum_sim import constants as C


# ------------------------------------------------------------------
# Node count:  R_{n,ℓ} has exactly (n − ℓ − 1) nodes
# ------------------------------------------------------------------

NODE_CASES = [
    (1, 0, 0),
    (2, 0, 1),
    (2, 1, 0),
    (3, 0, 2),
    (3, 1, 1),
    (3, 2, 0),
    (4, 0, 3),
    (4, 1, 2),
    (4, 2, 1),
    (4, 3, 0),
]


@pytest.mark.parametrize("n,ell,expected_nodes", NODE_CASES)
def test_node_count(n: int, ell: int, expected_nodes: int):
    """Count zero-crossings of R_{n,ℓ}(r) excluding r=0."""
    r = np.linspace(1e-15, (4 * n**2 + 20) * C.A0, 10_000)
    R = radial.R_nl(n, ell, r)
    # Count sign changes (nodes)
    signs = np.sign(R)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 1)
    assert sign_changes == expected_nodes, (
        f"R_{n},{ell}: expected {expected_nodes} nodes, found {sign_changes}"
    )


# ------------------------------------------------------------------
# Normalization: ∫₀^∞ |R_{n,ℓ}(r)|² r² dr = 1
# ------------------------------------------------------------------

NORM_CASES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 3)]


@pytest.mark.parametrize("n,ell", NORM_CASES)
def test_normalization(n: int, ell: int):
    r = np.linspace(0, (6 * n**2 + 50) * C.A0, 20_000)
    R = radial.R_nl(n, ell, r)
    integrand = R**2 * r**2
    norm = float(np.trapezoid(integrand, r))
    assert abs(norm - 1.0) < 1e-4, (
        f"R_{n},{ell}: norm = {norm:.8f}, expected 1.0"
    )


# ------------------------------------------------------------------
# Orthogonality: ⟨R_{n',ℓ} | R_{n,ℓ}⟩ = 0 for n' ≠ n
# ------------------------------------------------------------------

ORTHO_CASES = [
    (1, 0, 2, 0),
    (1, 0, 3, 0),
    (2, 0, 3, 0),
    (2, 1, 3, 1),
    (3, 1, 4, 1),
    (3, 2, 4, 2),
]


@pytest.mark.parametrize("n1,ell,n2,ell2", ORTHO_CASES)
def test_orthogonality(n1: int, ell: int, n2: int, ell2: int):
    assert ell == ell2  # orthogonality holds for same ℓ
    n_hi = max(n1, n2)
    r = np.linspace(0, (6 * n_hi**2 + 50) * C.A0, 20_000)
    R1 = radial.R_nl(n1, ell, r)
    R2 = radial.R_nl(n2, ell, r)
    integrand = R1 * R2 * r**2
    overlap = float(np.trapezoid(integrand, r))
    assert abs(overlap) < 1e-4, (
        f"⟨R_{n1},{ell}|R_{n2},{ell}⟩ = {overlap:.6e}, expected 0"
    )


# ------------------------------------------------------------------
# Model equivalence (isospectrality in eigenfunctions)
# ------------------------------------------------------------------

def test_rydberg_equals_dynamic_vacuum():
    """Both models produce identical R_{n,ℓ}(r)."""
    r = np.linspace(1e-15, 30 * C.A0, 500)
    for n, ell in [(1, 0), (2, 1), (3, 2)]:
        R_ryd = radial.R_nl(n, ell, r, model="rydberg")
        R_dv = radial.R_nl(n, ell, r, model="dynamic_vacuum")
        np.testing.assert_allclose(R_ryd, R_dv, rtol=1e-14)


# ------------------------------------------------------------------
# verify_orthonormality helper
# ------------------------------------------------------------------

def test_verify_orthonormality_all_pass():
    results = radial.verify_orthonormality(n_max=3, r_max_bohr=150, n_pts=10_000)
    for entry in results:
        assert entry["passed"], (
            f"Overlap ⟨{entry['n1']},{entry['ell']}|{entry['n2']},{entry['ell']}⟩"
            f" = {entry['overlap']:.6e}"
        )


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_invalid_quantum_numbers():
    r = np.linspace(0, 10 * C.A0, 100)
    with pytest.raises(ValueError):
        radial.R_nl(0, 0, r)  # n < 1
    with pytest.raises(ValueError):
        radial.R_nl(2, 2, r)  # ℓ ≥ n
    with pytest.raises(ValueError):
        radial.R_nl(1, -1, r)  # ℓ < 0


def test_invalid_model():
    r = np.linspace(0, 10 * C.A0, 100)
    with pytest.raises(ValueError):
        radial.R_nl(1, 0, r, model="unknown")
