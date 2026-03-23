"""dynamic_vacuum_sim — Hydrogenic spectra via Rydberg & dynamic-vacuum models.

Implements the standard Coulomb / Rydberg model alongside the dynamic-vacuum
acoustic framework of White et al., Phys. Rev. Research 8, 013264 (2026).

Quick start::

    from dynamic_vacuum_sim import rydberg, dynamic_vacuum, radial, plotting

    # Ground-state energy
    rydberg.level_energy(1)         # {'energy_eV': 13.598..., ...}

    # Dynamic-vacuum mapping (identical by isospectrality)
    dynamic_vacuum.level_energy(1)  # {'energy_eV': 13.598..., ...}

    # Verify exact match
    dynamic_vacuum.verify_isospectrality()

    # Radial wave function
    import numpy as np
    r = np.linspace(0, 20 * 5.295e-11, 500)
    R = radial.R_nl(2, 1, r)

    # Plot
    fig = plotting.plot_levels()
    fig.savefig("levels.png")
"""

__version__ = "0.1.0"
