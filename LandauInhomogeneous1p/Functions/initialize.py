"""Particle initialisation for the 1D-2V Landau damping test."""
import numpy as np


def landau_damping_density_x(x, par):
    return 1.0 + par.alpha * np.cos(par.k * x)


def landau_damping_f0(x, v1, v2, par):
    return landau_damping_density_x(x, par) * np.exp(-0.5 * (v1 * v1 + v2 * v2)) / (2.0 * np.pi)


def _cell_mass_x(i, par):
    a = i * par.eta
    b = (i + 1) * par.eta
    return (b - a) + par.alpha * (np.sin(par.k * b) - np.sin(par.k * a)) / par.k


def _sample_truncated_normal(rng, n, low, high):
    """Rejection sample N(0,1) restricted to [low,high]."""
    out = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        draw = rng.normal(size=max(4 * (n - filled), 128))
        draw = draw[(draw >= low) & (draw <= high)]
        take = min(draw.size, n - filled)
        if take > 0:
            out[filled:filled + take] = draw[:take]
            filled += take
    return out


def initialize_particles_landau_damping(par):
    """
    Combined sampling used for the Landau damping test:
    stratified in x, Gaussian in velocity.

    We keep exactly Nv1*Nv2*Nc particles in each spatial cell.  The particle
    weights carry the spatial factor 1+alpha*cos(kx), so sum(w)=Lx.
    """
    rng = np.random.default_rng(par.random_seed)
    Mx = par.particles_per_x_cell
    N = par.N

    x = np.empty(N, dtype=float)
    v1 = np.empty(N, dtype=float)
    v2 = np.empty(N, dtype=float)
    w = np.empty(N, dtype=float)

    pos = 0
    for i in range(par.Nx):
        sl = slice(pos, pos + Mx)
        # Stratified positions inside the cell reduce grid-scale noise.
        u = (np.arange(Mx, dtype=float) + rng.random(Mx)) / Mx
        rng.shuffle(u)
        x[sl] = (i + u) * par.eta

        mass_i = _cell_mass_x(i, par)
        w[sl] = mass_i / Mx
        pos += Mx

    if par.truncate_velocity_samples:
        v1[:] = _sample_truncated_normal(rng, N, -par.Lv1, par.Lv1)
        v2[:] = _sample_truncated_normal(rng, N, -par.Lv2, par.Lv2)
    else:
        v1[:] = rng.normal(size=N)
        v2[:] = rng.normal(size=N)

    if par.enforce_zero_momentum:
        mass = np.sum(w)
        v1 -= np.sum(w * v1) / mass
        v2 -= np.sum(w * v2) / mass

    x %= par.Lx
    return x, v1, v2, w


def make_batch_ids(N, par, rng):
    """Create equal-size random batches for the random-batch collision step."""
    if (not par.random_batch) or par.random_batches <= 1:
        return np.zeros(N, dtype=np.int64)
    R = par.random_batches
    if N % R != 0:
        raise ValueError("N must be divisible by random_batches")
    batch_ids = np.empty(N, dtype=np.int64)
    perm = rng.permutation(N)
    size = N // R
    for b in range(R):
        batch_ids[perm[b * size:(b + 1) * size]] = b
    return batch_ids
