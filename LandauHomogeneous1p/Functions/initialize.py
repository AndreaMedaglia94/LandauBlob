"""
BKW exact solution and particle initialization (two strategies).
"""
import numpy as np


def bkw_2d(t, T_tot, C_BKW, B_BKW, vx, vy):
    """Exact 2D BKW solution, evaluated elementwise at (vx, vy). m=1 fixed."""
    K = T_tot * (1 - C_BKW * np.exp(-B_BKW * 2 * t))
    norm = 1 / (2 * np.pi * K)
    c1 = (2 * K - T_tot) / K
    c2 = (T_tot - K) / (2 * K**2)
    r2 = vx**2 + vy**2
    return norm * np.exp(-r2 / (2 * K)) * (c1 + c2 * r2)


def initialize_particles_det_BKW(par):
    """
    One particle per cell, placed at the cell center, weighted by the
    BKW mass in that cell. par.grid_x/par.grid_y already ARE the mesh
    cell centers (see params.py), so at t=0 the particles coincide
    exactly with the error-reconstruction evaluation grid -- no need
    to rebuild the meshgrid here.
    """
    f0 = bkw_2d(0.0, par.temp, par.c, par.beta, par.grid_x, par.grid_y)
    wg = f0 * par.dV**2
    vx = par.grid_x.copy()
    vy = par.grid_y.copy()
    return vx, vy, wg


def initialize_particles_stratified_BKW(par, rng=None):
    """
    Stratified/jittered initialization: instead of ONE particle per cell
    sitting exactly at the cell center (with weight = local mass), place
    a NUMBER of equally-weighted particles per cell proportional to the
    local mass, each jittered to a uniformly random position within the
    cell. All particles end up with the same weight 1/N; the density
    information lives in particle COUNT/spacing instead of particle
    weight.
    """
    if rng is None:
        rng = np.random.default_rng()

    f0 = bkw_2d(0.0, par.temp, par.c, par.beta, par.grid_x, par.grid_y)
    massfraction = f0 * par.dV**2
    frac_per_cell = massfraction * par.N  # expected particle count per cell

    base_count = np.floor(frac_per_cell).astype(np.int64)
    remainder = frac_per_cell - base_count

    n_missing = par.N - base_count.sum()
    if n_missing > 0:
        probs = remainder / remainder.sum()
        idx_extra = rng.choice(len(probs), size=n_missing, replace=True, p=probs)
        np.add.at(base_count, idx_extra, 1)  # np.add.at (not +=) needed:
        # idx_extra can repeat (sampling WITH replacement), and a naive
        # base_count[idx_extra] += 1 silently drops repeats instead of
        # accumulating them.
    elif n_missing < 0:
        # Shouldn't happen: base_count is a floor, so sum(base_count) <= N
        # always. Guarding anyway rather than silently mis-assigning.
        raise RuntimeError(f"n_missing={n_missing} < 0; unexpected from floor-based base_count")

    final_count = base_count
    assert final_count.sum() == par.N, (
        f"Initialization error: assigned {final_count.sum()} particles, expected {par.N}"
    )

    # place final_count[c] particles uniformly at random inside cell c
    cell_x = np.repeat(par.grid_x, final_count)
    cell_y = np.repeat(par.grid_y, final_count)
    vx = cell_x + par.dV * (rng.random(par.N) - 0.5)
    vy = cell_y + par.dV * (rng.random(par.N) - 0.5)
    wg = np.full(par.N, 1.0 / par.N)

    return vx, vy, wg