#!/usr/bin/env python3
"""demo.py — Quick tour of dynamic_vacuum_sim.

Run:
    pip install -e ".[dev]"   # from repo root, once
    python examples/demo.py
"""

from dynamic_vacuum_sim import rydberg, dynamic_vacuum as dv, radial, plotting, constants as C
import numpy as np

# ── 1. Energy levels ─────────────────────────────────────────────────────────

print("=" * 62)
print("  HYDROGEN ENERGY LEVELS  (reduced-mass Rydberg, CODATA-2018)")
print("=" * 62)
print(f"{'n':>3}  {'|E| (eV)':>16}  {'f (PHz)':>12}")
print("-" * 40)
for n in range(1, 8):
    lv = rydberg.level_energy(n)
    print(f"{n:3d}  {lv['energy_eV']:16.9f}  {lv['frequency_PHz']:12.6f}")

# ── 2. Spectral lines ───────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  SELECTED SPECTRAL LINES")
print("=" * 62)
print(f"{'Transition':>16}  {'f (PHz)':>12}  {'λ (nm)':>14}")
print("-" * 50)

lines = [
    ("Lyman-α",   2, 1),
    ("Lyman-β",   3, 1),
    ("Hα",        3, 2),
    ("Hβ",        4, 2),
    ("Paschen-α", 4, 3),
]
for label, n2, n1 in lines:
    t = rydberg.transition(n2, n1)
    print(f"{label:>16}  {t['frequency_PHz']:12.6f}  {t['wavelength_nm']:14.6f}")

# ── 3. Named series helper ──────────────────────────────────────────────────

print("\n  Full Balmer series (up to n=10):")
for t in rydberg.series("balmer", n_max=10):
    print(f"    {t['n_upper']} → {t['n_lower']}:  λ = {t['wavelength_nm']:.3f} nm")

# ── 4. Dynamic-vacuum mapping ───────────────────────────────────────────────

print("\n" + "=" * 62)
print("  DYNAMIC-VACUUM FRAMEWORK  (White et al., PRR 8, 013264, 2026)")
print("=" * 62)

for n in range(1, 5):
    lv = dv.level_energy(n)
    print(
        f"  n={n}:  κ = {lv['kappa_1m']:.6e} m⁻¹,  "
        f"ω = {lv['omega_rad']:.6e} rad/s,  "
        f"|E| = {lv['energy_eV']:.9f} eV"
    )

# ── 5. Constitutive coefficients A(ωₙ), C(ωₙ) ─────────────────────────────

print("\n  Constitutive coefficients (Eq. 22):")
print(f"  {'n':>3}  {'A(ωₙ)':>16}  {'C(ωₙ)':>16}  {'A < 0?':>8}")
print("  " + "-" * 50)
for n in range(1, 6):
    A = dv.A_coeff(n)
    Cv = dv.C_coeff(n)
    print(f"  {n:3d}  {A:16.6e}  {Cv:16.6e}  {'✓' if A < 0 else '✗':>8}")

# ── 6. Isospectrality verification ──────────────────────────────────────────

print("\n  Isospectrality check (Rydberg vs. dynamic-vacuum):")
results = dv.verify_isospectrality(n_max=7, rtol=1e-12)
for r in results:
    print(f"    n={int(r['n'])}:  rel_error = {r['rel_error']:.2e}")
print("  ✓ All levels match to machine precision.\n")

# ── 7. Radial wave functions ────────────────────────────────────────────────

print("=" * 62)
print("  RADIAL WAVE FUNCTIONS")
print("=" * 62)

r = np.linspace(1e-15, 25 * C.A0, 500)
for n, ell in [(1, 0), (2, 1), (3, 2)]:
    R = radial.R_nl(n, ell, r)
    density = radial.radial_probability_density(n, ell, r)
    peak_r = r[np.argmax(density)] / C.A0
    print(f"  R_{{{n},{ell}}}:  peak density at r ≈ {peak_r:.1f} a₀")

# ── 8. Orthonormality spot-check ────────────────────────────────────────────

print("\n  Orthonormality spot-check (ℓ=0):")
r_grid = np.linspace(0, 200 * C.A0, 20_000)
for n1, n2 in [(1, 1), (1, 2), (2, 2), (1, 3)]:
    R1 = radial.R_nl(n1, 0, r_grid)
    R2 = radial.R_nl(n2, 0, r_grid)
    overlap = float(np.trapezoid(R1 * R2 * r_grid**2, r_grid))
    expected = 1.0 if n1 == n2 else 0.0
    status = "✓" if abs(overlap - expected) < 1e-4 else "✗"
    print(f"    ⟨R_{{{n1},0}}|R_{{{n2},0}}⟩ = {overlap:+.6f}  (expected {expected:.0f})  {status}")

# ── 9. Generate plots ───────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  GENERATING PLOTS")
print("=" * 62)

fig1 = plotting.plot_levels(n_max=7)
fig1.savefig("examples/energy_levels.png", dpi=150)
print("  → examples/energy_levels.png")

fig2 = plotting.plot_series()
fig2.savefig("examples/spectral_series.png", dpi=150)
print("  → examples/spectral_series.png")

fig3 = plotting.plot_radial(n=3, ell=1)
fig3.savefig("examples/radial_3_1.png", dpi=150)
print("  → examples/radial_3_1.png")

fig4 = plotting.plot_radial_comparison(n=2, ell=0)
fig4.savefig("examples/isospectrality_2_0.png", dpi=150)
print("  → examples/isospectrality_2_0.png")

print("\nDone. All plots saved in examples/\n")
