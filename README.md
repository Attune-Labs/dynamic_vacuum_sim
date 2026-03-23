# Dynamic Vacuum Sim

**dynamic\_vacuum\_sim** computes hydrogenic energy levels, transition lines, and radial eigenfunctions using both the standard Coulomb–Schrödinger (Rydberg) model and the dynamic-vacuum acoustic framework introduced by White, Vera, Sylvester & Dudzinski in [*Phys. Rev. Research* **8**, 013264 (2026)](https://doi.org/10.1103/l8y7-r3rm). The package verifies exact isospectrality between the two models — the 1/n² Rydberg ladder is reproduced to machine precision with a single reduced-mass calibration and zero free parameters.

This lets you compare the standard quantum-mechanical and dynamic-vacuum pictures with a single API and shared test suite.

## Who is this for?

| Audience | Use case |
|----------|----------|
| **Physics grad students & PIs** | Reproduce White et al. results; explore constitutive coefficients A(ω\_n), C(ω\_n) and the Madelung dispersion; use as a verified Rydberg-ladder reference implementation. |
| **Analog-quantum-simulator & metamaterials groups** | Extract the frequency-dependent constitutive profile 1/c²\_s(r) = A + C/r for designing acoustic or photonic analog hydrogen resonators. |
| **Advanced educators** | Demonstrate the link between dispersive continuum mechanics and quantum mechanics in a graduate course; every function cites numbered equations in the paper. |

## Quickstart

### Install

```bash
pip install dynamic_vacuum_sim
```

For an editable/developer install (includes test dependencies):

```bash
git clone https://github.com/Attune-Labs/dynamic_vacuum_sim.git
cd dynamic_vacuum_sim
pip install -e ".[dev]"
```

### Rydberg model — energy levels and spectral lines

```python
from dynamic_vacuum_sim import rydberg

# Energy levels for n = 1, 2, 3
for n in range(1, 4):
    lv = rydberg.level_energy(n)
    print(f"n={n}:  |E| = {lv['energy_eV']:.9f} eV,  f = {lv['frequency_PHz']:.6f} PHz")
# n=1:  |E| = 13.598287264 eV,  f = 3.288051 PHz
# n=2:  |E| =  3.399571816 eV,  f = 0.822013 PHz
# n=3:  |E| =  1.510920807 eV,  f = 0.365339 PHz

# H-alpha (Balmer series, 3 → 2)
ha = rydberg.transition(n_upper=3, n_lower=2)
print(f"H-α:  λ = {ha['wavelength_nm']:.3f} nm,  f = {ha['frequency_PHz']:.6f} PHz")
# H-α:  λ = 656.470 nm,  f = 0.456674 PHz
```

### Dynamic-vacuum model — verify isospectrality

```python
from dynamic_vacuum_sim import rydberg, dynamic_vacuum as dv

# Both models return the same energy for any n
e_ryd = rydberg.level_energy(3)["energy_eV"]       # 1.510920807 eV
e_dv  = dv.level_energy(3)["energy_eV"]             # 1.510920807 eV
assert e_ryd == e_dv  # True — bit-exact match

# The DV model also exposes the acoustic quantities
lv = dv.level_energy(3)
print(f"κ₃ = {lv['kappa_1m']:.6e} m⁻¹")   # bound-state inverse length
print(f"ω₃ = {lv['omega_rad']:.6e} rad/s")  # eigenfrequency

# Batch verification for n = 1 … 20, tolerance 1e-12
dv.verify_isospectrality(n_max=20, rtol=1e-12)  # passes silently
```

### Command-line interface

```bash
dv-hydrogen level --n 1 --n 2 --n 3              # print energy levels
dv-hydrogen line  --n1 2 --n2 5                   # transition (upper=5 → lower=2)
dv-hydrogen verify --nmax 7                       # isospectrality check
dv-hydrogen plot radial --n 3 --ell 2 -o R32.png  # save a radial-density plot
dv-hydrogen plot levels --nmax 7 -o levels.png    # energy-level ladder diagram
dv-hydrogen plot series -o series.png             # Lyman / Balmer / Paschen series
```

## Theory at a glance

The package implements two mathematically equivalent descriptions of the hydrogen atom.

### 1. Standard Rydberg model (`rydberg.py`, `radial.py`)

Energy levels and transition frequencies use the reduced-mass Rydberg constant R\_H (CODATA-2018):

`|E_n| = h c R_H / n²` and `f(n₂→n₁) = c R_H (1/n₁² − 1/n₂²)`

(See Eqs. 19–20 in White et al.)

Radial eigenfunctions use the normalized Laguerre form:

`R_nℓ(r) = N_nℓ (2κr)^ℓ exp(−κr) L^(2ℓ+1)_(n−ℓ−1)(2κr)`

(See Eq. 13a in White et al.)

These are exposed by `rydberg.level_energy()`, `rydberg.transition()`, `rydberg.series()`, `radial.R_nl()`, and `radial.radial_probability_density()`.

### 2. Dynamic-vacuum acoustic model (`dynamic_vacuum.py`)

White et al. show that a dispersive compressible vacuum with quadratic dispersion and a Coulombic constitutive profile produces the same spectrum:

`ω = D q²` with `D = ℏ/(2μ)` — quadratic dispersion (Eq. 1, 11)

`k²_eff(r; ω) = α(ω) + β(ω)/r` with `β = 2/a₀` — Coulombic profile (Eq. 7, 18)

Bound-state quantization then gives:

`κ_n = β/(2n)`, `ω_n = D κ_n² = ω*/n²`, `E_n = −ℏ ω_n`

(See Eqs. 10, 18–19 in White et al.)

Calibrating to the reduced-mass Rydberg frequency `ω* = 2πcR_H` fixes `D = ℏ/(2μ)` and `m_eff = μ` with no free parameters.

These are exposed by `dynamic_vacuum.kappa_n()`, `dynamic_vacuum.omega_n()`, `dynamic_vacuum.level_energy()`, `dynamic_vacuum.k_eff_squared()`, and `dynamic_vacuum.verify_isospectrality()`.

**Advanced details.** The frequency-dependent constitutive coefficients `A(ω_n) = −n²/(a₀² ω*²)` and `C(ω_n) = 2n⁴/(a₀ ω*²)` (Eq. 22) are available via `dynamic_vacuum.A_coeff()` and `dynamic_vacuum.C_coeff()`. `A < 0` identifies the reactive stop band; `C > 0` represents proton-induced coupling. The full Madelung dispersion `ω² = c_L² k² + D² k⁴` (Eq. A21) is available via `dynamic_vacuum.dispersion_full()`.

### What is implemented vs. what is not

| Status | Feature | Code |
|--------|---------|------|
| Implemented | 1/n² energy levels and transition frequencies | `rydberg.level_energy()`, `rydberg.transition()`, `rydberg.series()` |
| Implemented | Radial eigenfunctions R\_nℓ(r) and probability densities | `radial.R_nl()`, `radial.radial_probability_density()` |
| Implemented | Orthonormality verification | `radial.verify_orthonormality()` |
| Implemented | Dynamic-vacuum eigenvalues κ\_n, ω\_n, E\_n | `dynamic_vacuum.kappa_n()`, `.omega_n()`, `.level_energy()` |
| Implemented | Constitutive coefficients A(ω\_n), C(ω\_n) | `dynamic_vacuum.A_coeff()`, `dynamic_vacuum.C_coeff()` |
| Implemented | Effective wave number k²\_eff(r) | `dynamic_vacuum.k_eff_squared()` |
| Implemented | Full Madelung dispersion (Eq. A21) | `dynamic_vacuum.dispersion_full()` |
| Implemented | Isospectrality verification (Rydberg vs. DV) | `dynamic_vacuum.verify_isospectrality()` |
| Implemented | Plotting (level ladders, series diagrams, radial densities, model comparison) | `plotting.plot_levels()`, `.plot_series()`, `.plot_radial()`, `.plot_radial_comparison()` |
| Implemented | CLI for levels, lines, verification, and plot generation | `cli.py` → `dv-hydrogen` |
| Not implemented | Isotope shifts (μ → μ(M)) | Discussed in Sec. V of the paper |
| Not implemented | Stark/Zeeman perturbative analogues | Discussed in Sec. V of the paper |
| Not implemented | Multi-center problems | Discussed in Sec. V of the paper |

## Package structure

```
dynamic_vacuum_sim/
├── constants.py        # CODATA-2018 constants, reduced-mass Rydberg quantities
├── rydberg.py          # Standard 1/n² Rydberg model
├── dynamic_vacuum.py   # Dynamic-vacuum mapping (White et al. 2026)
├── radial.py           # Hydrogenic radial eigenfunctions R_{nℓ}(r)
├── plotting.py         # Matplotlib helpers (levels, series, radial densities)
└── cli.py              # Click-based CLI (dv-hydrogen)
```

## Examples

The [`examples/`](examples/) directory contains ready-to-run demos:

| File | Description |
|------|-------------|
| [`quickstart.ipynb`](examples/quickstart.ipynb) | Jupyter notebook covering every feature with inline plots |
| [`demo.py`](examples/demo.py) | Standalone script — run `python examples/demo.py` for a full terminal tour |

## Running tests

```bash
pytest -v    # 81 tests: levels, lines, isospectrality, radial nodes,
             # normalization, orthogonality, constitutive coefficients, edge cases
```

## Constants

All fundamental constants use CODATA-2018 values from Tiesinga et al., *Rev. Mod. Phys.* **93**, 025010 (2021). The reduced-mass Rydberg constant R\_H = R\_∞ · μ/m\_e ensures agreement with the hydrogen spectrum (not the infinite-mass limit).

## Citing

If you use this package in published work, please cite both the software and the underlying paper.

**This package:**

```bibtex
@software{dynamic_vacuum_sim,
  author       = {Attune-Labs},
  title        = {dynamic\_vacuum\_sim: Hydrogenic spectra via Rydberg and
                  dynamic-vacuum models},
  version      = {0.1.0},
  year         = {2026},
  url          = {https://github.com/Attune-Labs/dynamic_vacuum_sim},
}
```

**White et al. (2026):**

```bibtex
@article{White2026emergent,
  author    = {White, Harold and Vera, Jerry and Sylvester, Andre
               and Dudzinski, Leonard},
  title     = {Emergent quantization from a dynamic vacuum},
  journal   = {Physical Review Research},
  volume    = {8},
  number    = {1},
  pages     = {013264},
  year      = {2026},
  doi       = {10.1103/l8y7-r3rm},
  publisher = {American Physical Society},
}
```

## License

MIT
