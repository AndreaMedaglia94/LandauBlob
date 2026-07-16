"""Particle initialisation for the two-species 1D-2V Landau damping test."""
import numpy as np


def _density_x(x, alpha, par):
    return 1.0 + alpha * np.cos(par.k * x)


def _cell_mass_x(i, alpha, density, par):
    a = i * par.eta
    b = (i + 1) * par.eta
    return density * ((b - a) + alpha * (np.sin(par.k * b) - np.sin(par.k * a)) / par.k)


def _sample_truncated_normal(rng, n, mean, std, low, high):
    if std == 0.0:
        out = np.full(n, mean, dtype=float)
        if np.any(out < low) or np.any(out > high):
            raise ValueError("zero-temperature initial velocity lies outside the truncation interval")
        return out

    out = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        draw = rng.normal(loc=mean, scale=std, size=max(4 * (n - filled), 256))
        draw = draw[(draw >= low) & (draw <= high)]
        take = min(draw.size, n - filled)
        if take > 0:
            out[filled:filled + take] = draw[:take]
            filled += take
    return out


def _initialise_one_species(rng, N, Mx, alpha, density, temp, mass, u1, u2, Lv1, Lv2, par):
    x = np.empty(N, dtype=float)
    v1 = np.empty(N, dtype=float)
    v2 = np.empty(N, dtype=float)
    w = np.empty(N, dtype=float)

    pos = 0
    for i in range(par.Nx):
        sl = slice(pos, pos + Mx)
        u = (np.arange(Mx, dtype=float) + rng.random(Mx)) / Mx
        rng.shuffle(u)
        x[sl] = (i + u) * par.eta
        w[sl] = _cell_mass_x(i, alpha, density, par) / Mx
        pos += Mx

    std = np.sqrt(temp / mass) if temp > 0.0 else 0.0
    if par.truncate_velocity_samples:
        v1[:] = _sample_truncated_normal(rng, N, u1, std, -Lv1, Lv1)
        v2[:] = _sample_truncated_normal(rng, N, u2, std, -Lv2, Lv2)
    else:
        v1[:] = rng.normal(loc=u1, scale=std, size=N) if std > 0.0 else u1
        v2[:] = rng.normal(loc=u2, scale=std, size=N) if std > 0.0 else u2

    if par.enforce_zero_species_mean_velocity:
        ww = np.sum(w)
        v1 += u1 - np.sum(w * v1) / ww
        v2 += u2 - np.sum(w * v2) / ww

    x %= par.Lx
    return x, v1, v2, w


def initialize_particles_landau_damping_2species(par):
    """
    Stratified spatial sampling and Maxwellian velocity sampling for two species.

    Species 1 carries alpha1 by default.  Species 2 carries alpha2, which is zero
    by default, representing initially uniform ions.
    """
    rng = np.random.default_rng(par.random_seed)

    x1, v11, v12, w1 = _initialise_one_species(
        rng, par.N1, par.particles_per_x_cell_1,
        par.alpha1, par.n1, par.temp1, par.m1,
        par.u1_v1_0, par.u1_v2_0, par.Lv1_v1, par.Lv1_v2, par
    )
    x2, v21, v22, w2 = _initialise_one_species(
        rng, par.N2, par.particles_per_x_cell_2,
        par.alpha2, par.n2, par.temp2, par.m2,
        par.u2_v1_0, par.u2_v2_0, par.Lv2_v1, par.Lv2_v2, par
    )

    if par.enforce_zero_total_current:
        # Remove the residual mean current in the v1 direction.  This is useful
        # for the Ampere Landau-damping test, where the initial current should be zero.
        c1 = par.current_factor1
        c2 = par.current_factor2
        denom = c1 * np.sum(w1) + c2 * np.sum(w2)
        # If the charge-weighted denominator is near zero, remove current by changing
        # only species 1, which avoids a singular global correction in neutral cases.
        J1 = c1 * np.sum(w1 * v11) + c2 * np.sum(w2 * v21)
        J2 = c1 * np.sum(w1 * v12) + c2 * np.sum(w2 * v22)
        if abs(c1) > 0.0:
            v11 -= J1 / (c1 * np.sum(w1))
            v12 -= J2 / (c1 * np.sum(w1))
        elif abs(denom) > 1.0e-14:
            shift1 = J1 / denom
            shift2 = J2 / denom
            v11 -= shift1
            v12 -= shift2
            v21 -= shift1
            v22 -= shift2

    return x1, v11, v12, w1, x2, v21, v22, w2


def make_batch_ids(N, par, rng):
    """Create equal-size random batches for one species."""
    if (not par.random_batch) or par.random_batches <= 1:
        return np.zeros(N, dtype=np.int64)
    R = par.random_batches
    if N % R != 0:
        raise ValueError("species particle count must be divisible by random_batches")
    batch_ids = np.empty(N, dtype=np.int64)
    perm = rng.permutation(N)
    size = N // R
    for b in range(R):
        batch_ids[perm[b * size:(b + 1) * size]] = b
    return batch_ids

class BatchIdGenerator:
    """Reusable random-batch id generator.

    This avoids allocating a fresh permutation and batch-id array at every time
    step.  Re-shuffling an existing permutation is still a uniform random
    permutation, and the same equal-size batch construction is used by
    make_batch_ids.
    """
    def __init__(self, N, par, rng):
        self.N = int(N)
        self.random_batch = bool(par.random_batch and par.random_batches > 1)
        self.R = int(par.random_batches) if self.random_batch else 1
        self.rng = rng
        self.batch_ids = np.zeros(self.N, dtype=np.int64)
        self.base_perm = np.arange(self.N, dtype=np.int64)
        self.perm = self.base_perm.copy()
        if self.random_batch:
            if self.N % self.R != 0:
                raise ValueError("species particle count must be divisible by random_batches")
            self.size = self.N // self.R
        else:
            self.size = self.N

    def next(self):
        if not self.random_batch:
            return self.batch_ids
        # Reset to arange before shuffling.  This matches rng.permutation(N)
        # exactly while avoiding per-step allocation of the arrays.
        self.perm[:] = self.base_perm
        self.rng.shuffle(self.perm)
        size = self.size
        for b in range(self.R):
            self.batch_ids[self.perm[b * size:(b + 1) * size]] = b
        return self.batch_ids

