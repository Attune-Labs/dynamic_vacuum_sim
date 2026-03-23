# dynamic_vacuum_sim

Hydrogenic spectra computed via the standard Coulomb/Rydberg model **and** the dynamic-vacuum acoustic framework from:

> Harold White, Jerry Vera, Andre Sylvester & Leonard Dudzinski,  
> *"Emergent quantization from a dynamic vacuum"*,  
> Phys. Rev. Research **8**, 013264 (2026).  
> DOI: [10.1103/l8y7-r3rm](https://doi.org/10.1103/l8y7-r3rm)

The package demonstrates exact isospectrality between the two models: a single reduced-mass calibration (`D = ℏ/(2μ)`, `β = 2/a₀`) reproduces the full hydrogen spectrum with no free parameters.

## Installation

```bash
git clone https://github.com/Attune-Labs/dynamic_vacuum_sim.git
cd dynamic_vacuum_sim
pip install -e ".[dev]"
```

## Quick start

### Python API

```python
from dynamic_vacuum_sim import rydberg, dynamic_vacuum, radial, plotting
import numpy as np

# Ground-state energy (reduced-mass Rydberg)
rydberg.level_energy(1)
# {'n': 1, 'energy_eV': 13.598434..., 'frequency_PHz': 3.288051..., ...}

# Dynamic-vacuum mapping — identical result
dynamic_vacuum.level_energy(1)
# {'n': 1, 'energy_eV': 13.598434..., 'kappa_1m': ..., 'omega_rad': ..., ...}

# Verify isospectrality to machine precision
dynamic_vacuum.verify_isospectrality(n_max=7, rtol=1e-12)

# Lyman-α transition
rydberg.transition(n_upper=2, n_lower=1)
# {'frequency_PHz': 2.466038..., 'wavelength_nm': 121.568446..., ...}

# Radial wave function R_{3,2}(r)
from dynamic_vacuum_sim.constants import A0
r = np.linspace(0, 40 * A0, 1000)
R = radial.R_nl(3, 2, r)

# Plots
fig = plotting.plot_levels(n_max=7)
fig.savefig("levels.png")

fig = plotting.plot_radial(n=3, ell=2)
fig.savefig("radial_3_2.png")

fig = plotting.plot_radial_comparison(n=3, ell=1)
fig.savefig("comparison_3_1.png")
```

### Command-line interface

```bash
# Energy levels
dv-hydrogen level --n 1 --n 2 --n 3 --n 4 --n 5 --n 6 --n 7

# Specific transition
dv-hydrogen line --n1 2 --n2 5

# Isospectrality verification
dv-hydrogen verify --nmax 7

# Plots
dv-hydrogen plot levels --nmax 7 -o levels.png
dv-hydrogen plot radial --n 3 --ell 2 --model dynamic_vacuum -o radial.png
dv-hydrogen plot series -o series.png
```

## Package structure

```
dynamic_vacuum_sim/
├── constants.py        # CODATA-2018 constants, reduced-mass Rydberg quantities
├── rydberg.py          # Standard 1/n² Rydberg model
├── dynamic_vacuum.py   # Dynamic-vacuum mapping (White et al. 2026)
├── radial.py           # Hydrogenic radial eigenfunctions R_{nℓ}(r)
├── plotting.py         # Matplotlib helpers (levels, series, radial densities)
└── cli.py              # Click-based CLI (`dv-hydrogen`)
```

## Key equations (paper references)

| Equation | Paper ref | Code |
|----------|-----------|------|
| ω = D q², D = ℏ/(2m_eff) | Eq. (1) | `constants.D_DISP` |
| k²_eff = α + β/r | Eq. (7) | `dynamic_vacuum.k_eff_squared()` |
| κ_n = β/(2n) = 1/(n a₀) | Eq. (18) | `dynamic_vacuum.kappa_n()` |
| ω_n = D κ_n² = ω*/n² | Eq. (10) | `dynamic_vacuum.omega_n()` |
| E_n = −ℏ ω_n | Eq. (19) | `dynamic_vacuum.level_energy()` |
| f = c R_H (1/n₁² − 1/n₂²) | Eq. (20) | `rydberg.transition()` |
| A(ω_n) = −n²/(a₀² ω*²) | Eq. (22) | `dynamic_vacuum.A_coeff()` |
| C(ω_n) = 2n⁴/(a₀ ω*²) | Eq. (22) | `dynamic_vacuum.C_coeff()` |
| R_{nℓ}(r) via Laguerre | Eq. (13a) | `radial.R_nl()` |
| ω² = c_L² k² + D² k⁴ | Eq. (A21) | `dynamic_vacuum.dispersion_full()` |

## Running tests

```bash
pytest -v
```

## Constants

All fundamental constants use CODATA-2018 values from Tiesinga et al., *Rev. Mod. Phys.* **93**, 025010 (2021). The reduced-mass Rydberg constant `R_H = R_∞ · μ/m_e` ensures agreement with the hydrogen spectrum (not the infinite-mass limit).

## License

MIT
