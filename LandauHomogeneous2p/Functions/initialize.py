"""Exact solutions and particle initialization for the two-species code."""
import numpy as np


def bkw_2d(t, temp, C_BKW, beta_BKW, vx, vy, mass, density):
    """Two-dimensional multispecies BKW solution used in the MATLAB code."""
    K = temp * (1.0 - C_BKW * np.exp(-2.0 * beta_BKW * t))
    r2 = vx**2 + vy**2
    norm = density * mass / (2.0 * np.pi * K)
    c1 = (2.0 * K - temp) / K
    c2 = mass * (temp - K) / (2.0 * K**2)
    return norm * np.exp(-mass * r2 / (2.0 * K)) * (c1 + c2 * r2)


def maxwellian_2d(vx, vy, density, mass, ux, uy, temp):
    """Two-dimensional Maxwellian density."""
    r2 = (vx - ux) ** 2 + (vy - uy) ** 2
    return density * mass / (2.0 * np.pi * temp) * np.exp(-mass * r2 / (2.0 * temp))


def _rng(par, rng=None):
    if rng is not None:
        return rng
    return np.random.default_rng(par.random_seed)


def _rescale_species_to_moments(vx, vy, wg, mass, density, target_u=(0.0, 0.0), target_temp=1.0):
    """Shift and isotropically scale one species to the requested mean and temperature."""
    ux = np.sum(wg * vx) / density
    uy = np.sum(wg * vy) / density
    thermal = mass * np.sum(wg * ((vx - ux) ** 2 + (vy - uy) ** 2)) / (2.0 * density)
    if thermal <= 0.0:
        raise ValueError("cannot rescale species with non-positive temperature")
    scale = np.sqrt(target_temp / thermal)
    vx = target_u[0] + scale * (vx - ux)
    vy = target_u[1] + scale * (vy - uy)
    return vx, vy


def initialize_particles_det_BKW(par, species):
    """One particle per cell, placed at the grid cell center."""
    grid_x, grid_y, dV, _, _, mass, density = par.species_grid(species)
    f0 = bkw_2d(0.0, par.temp, par.c, par.beta, grid_x, grid_y, mass, density)
    wg = f0 * dV**2
    vx = grid_x.copy()
    vy = grid_y.copy()
    return vx, vy, wg


def initialize_particles_stratified_BKW(par, species, rng=None):
    """
    Stratified particle initialization for a BKW profile.

    Particles have equal weights n_i/N.  Cell counts are sampled from the
    grid-cell probability masses.  This keeps the total species mass exactly
    equal to n_i even when the finite velocity box truncates a small tail.
    """
    rng = _rng(par, rng)
    grid_x, grid_y, dV, _, _, mass, density = par.species_grid(species)
    f0 = bkw_2d(0.0, par.temp, par.c, par.beta, grid_x, grid_y, mass, density)

    cell_mass = np.maximum(f0 * dV**2, 0.0)
    total_cell_mass = cell_mass.sum()
    if total_cell_mass <= 0.0:
        raise ValueError("initial BKW density has zero discrete mass on the chosen grid")

    expected_count = par.N * cell_mass / total_cell_mass
    base_count = np.floor(expected_count).astype(np.int64)
    remainder = expected_count - base_count
    n_missing = par.N - int(base_count.sum())

    if n_missing > 0:
        if remainder.sum() > 0.0:
            probs = remainder / remainder.sum()
        else:
            probs = cell_mass / cell_mass.sum()
        idx_extra = rng.choice(len(probs), size=n_missing, replace=True, p=probs)
        np.add.at(base_count, idx_extra, 1)
    elif n_missing < 0:
        # This should not occur with floor counts, but keep a guard for numerical issues.
        idx = np.argsort(remainder)[: -n_missing]
        base_count[idx] -= 1

    if int(base_count.sum()) != par.N:
        raise RuntimeError(f"assigned {base_count.sum()} particles, expected {par.N}")

    cell_x = np.repeat(grid_x, base_count)
    cell_y = np.repeat(grid_y, base_count)
    vx = cell_x + dV * (rng.random(par.N) - 0.5)
    vy = cell_y + dV * (rng.random(par.N) - 0.5)
    wg = np.full(par.N, density / par.N)

    if par.enforce_initial_moments:
        vx, vy = _rescale_species_to_moments(vx, vy, wg, mass, density, (0.0, 0.0), par.temp)

    return vx, vy, wg


def initialize_particles_gaussian(par, species, rng=None):
    """Random Maxwellian initialization, useful for Coulomb relaxation tests."""
    rng = _rng(par, rng)
    if species == 1:
        mass, density = par.m1, par.n1
        ux, uy, temp = par.u_1_x_0, par.u_1_y_0, par.temp_1_0
    elif species == 2:
        mass, density = par.m2, par.n2
        ux, uy, temp = par.u_2_x_0, par.u_2_y_0, par.temp_2_0
    else:
        raise ValueError("species must be 1 or 2")

    std = np.sqrt(temp / mass)
    vx = ux + std * rng.standard_normal(par.N)
    vy = uy + std * rng.standard_normal(par.N)
    wg = np.full(par.N, density / par.N)

    if par.enforce_initial_moments:
        vx, vy = _rescale_species_to_moments(vx, vy, wg, mass, density, (ux, uy), temp)

    return vx, vy, wg
