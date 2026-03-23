"""Command-line interface for ``dv-hydrogen``.

Usage examples::

    dv-hydrogen level --n 1 --n 7
    dv-hydrogen line  --n1 1 --n2 3
    dv-hydrogen verify --nmax 7
    dv-hydrogen plot radial  --n 3 --ell 2 --model rydberg -o radial_3_2.png
    dv-hydrogen plot levels  --nmax 7 -o levels.png
    dv-hydrogen plot series  -o series.png
"""

from __future__ import annotations

import json
import sys

import click

from . import rydberg
from . import dynamic_vacuum as dv


@click.group()
@click.version_option(package_name="dynamic_vacuum_sim")
def main() -> None:
    """dv-hydrogen: Hydrogenic spectra via Rydberg & dynamic-vacuum models."""


# -----------------------------------------------------------------------
# level
# -----------------------------------------------------------------------

@main.command()
@click.option(
    "--n",
    "ns",
    multiple=True,
    type=int,
    required=True,
    help="Principal quantum number(s).  Repeat for multiple: --n 1 --n 7",
)
@click.option(
    "--model",
    type=click.Choice(["rydberg", "dynamic_vacuum"], case_sensitive=False),
    default="rydberg",
    show_default=True,
    help="Spectral model.",
)
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
def level(ns: tuple[int, ...], model: str, json_output: bool) -> None:
    """Print energy level(s) for given principal quantum number(s)."""
    results = []
    for n in ns:
        if model == "dynamic_vacuum":
            d = dv.level_energy(n)
        else:
            d = rydberg.level_energy(n)
        results.append(d)

    if json_output:
        click.echo(json.dumps(results, indent=2))
    else:
        click.echo(f"{'n':>3}  {'|E| (eV)':>18}  {'f (PHz)':>14}  model={model}")
        click.echo("-" * 55)
        for d in results:
            click.echo(
                f"{d['n']:3d}  {d['energy_eV']:18.9f}  {d['frequency_PHz']:14.6f}"
            )


# -----------------------------------------------------------------------
# line
# -----------------------------------------------------------------------

@main.command()
@click.option("--n1", type=int, required=True, help="Lower level.")
@click.option("--n2", type=int, required=True, help="Upper level.")
@click.option("--json-output", is_flag=True, help="Emit JSON.")
def line(n1: int, n2: int, json_output: bool) -> None:
    """Print transition frequency and wavelength."""
    d = rydberg.transition(n2, n1)
    if json_output:
        click.echo(json.dumps(d, indent=2))
    else:
        click.echo(f"Transition  {n2} → {n1}")
        click.echo(f"  Frequency : {d['frequency_PHz']:.6f} PHz")
        click.echo(f"  Wavelength: {d['wavelength_nm']:.6f} nm")


# -----------------------------------------------------------------------
# verify
# -----------------------------------------------------------------------

@main.command()
@click.option("--nmax", type=int, default=7, show_default=True)
@click.option("--rtol", type=float, default=1e-12, show_default=True)
def verify(nmax: int, rtol: float) -> None:
    """Run isospectrality check (dynamic-vacuum vs. Rydberg)."""
    try:
        results = dv.verify_isospectrality(n_max=nmax, rtol=rtol)
    except AssertionError as exc:
        click.echo(f"FAIL: {exc}", err=True)
        sys.exit(1)

    click.echo(f"{'n':>3}  {'E_rydberg (eV)':>18}  {'E_dv (eV)':>18}  {'rel_err':>12}")
    click.echo("-" * 60)
    for r in results:
        click.echo(
            f"{int(r['n']):3d}  {r['E_rydberg_eV']:18.9f}  "
            f"{r['E_dv_eV']:18.9f}  {r['rel_error']:12.2e}"
        )
    click.echo(f"\nPASS — all n=1..{nmax} within rtol={rtol:.0e}")


# -----------------------------------------------------------------------
# plot  (sub-group)
# -----------------------------------------------------------------------

@main.group()
def plot() -> None:
    """Generate and save plots."""


@plot.command("radial")
@click.option("--n", type=int, required=True)
@click.option("--ell", type=int, required=True)
@click.option(
    "--model",
    type=click.Choice(["rydberg", "dynamic_vacuum"], case_sensitive=False),
    default="rydberg",
    show_default=True,
)
@click.option("-o", "--output", type=click.Path(), default=None)
def plot_radial_cmd(n: int, ell: int, model: str, output: str | None) -> None:
    """Plot radial probability density for (n, ℓ)."""
    from . import plotting

    fig = plotting.plot_radial(n, ell, model=model)
    out = output or f"radial_n{n}_l{ell}_{model}.png"
    fig.savefig(out, dpi=150)
    click.echo(f"Saved → {out}")


@plot.command("levels")
@click.option("--nmax", type=int, default=7, show_default=True)
@click.option(
    "--model",
    type=click.Choice(["rydberg", "dynamic_vacuum"], case_sensitive=False),
    default="rydberg",
    show_default=True,
)
@click.option("-o", "--output", type=click.Path(), default=None)
def plot_levels_cmd(nmax: int, model: str, output: str | None) -> None:
    """Plot energy level ladder diagram."""
    from . import plotting

    fig = plotting.plot_levels(n_max=nmax, model=model)
    out = output or f"levels_{model}.png"
    fig.savefig(out, dpi=150)
    click.echo(f"Saved → {out}")


@plot.command("series")
@click.option("--nmax", type=int, default=7, show_default=True)
@click.option("-o", "--output", type=click.Path(), default=None)
def plot_series_cmd(nmax: int, output: str | None) -> None:
    """Plot spectral series (Lyman, Balmer, Paschen)."""
    from . import plotting

    fig = plotting.plot_series(n_max=nmax)
    out = output or "spectral_series.png"
    fig.savefig(out, dpi=150)
    click.echo(f"Saved → {out}")
