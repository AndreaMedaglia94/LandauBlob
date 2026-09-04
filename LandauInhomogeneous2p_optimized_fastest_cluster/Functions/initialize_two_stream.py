"""
Particle initialisation for the two-stream instability, 1D-2V,
"""
import numpy as np


_N_STRAT_CELLS = 200  # Fixed spatial stratification resolution, matching
# the MATLAB reference exactly. Independent of par.Nx (the field/deposition
# grid) by design -- this is purely how initial positions are drawn.


def _stratified_positions(N, alpha, Lx, k, rng):
    """N positions in [0, Lx), stratified so the particle COUNT per cell
    (not the per-particle weight) follows 1 + alpha*cos(k*x)."""
    XX = np.linspace(0.0, Lx, _N_STRAT_CELLS + 1)
    XXm = 0.5 * (XX[1:] + XX[:-1])
    dXm = XX[1] - XX[0]
    fx = 1.0 + alpha * np.cos(k * XXm)
    # (2*pi/k) is one full wavelength; with Lx == 2*pi/k (as set in
    # Params.__post_init__), cell_area sums to ~1 over the domain, so
    # cell_area*N sums to ~N -- matching the reference's normalisation.
    cell_area = fx * dXm / (2.0 * np.pi / k)
    target = cell_area * N
    N_cell = np.maximum(np.round(target), 0.0).astype(np.int64)

    deficit = N - int(N_cell.sum())
    if deficit != 0:
        remainder = target - N_cell
        order = np.argsort(-remainder) if deficit > 0 else np.argsort(remainder)
        step = 1 if deficit > 0 else -1
        for idx in order[:abs(deficit)]:
            N_cell[idx] += step

    x = np.empty(N, dtype=float)
    pos = 0
    for i in range(_N_STRAT_CELLS):
        n_i = int(N_cell[i])
        if n_i == 0:
            continue
        x[pos:pos + n_i] = XX[i] + (XX[i + 1] - XX[i]) * rng.random(n_i)
        pos += n_i

    rng.shuffle(x)
    return x


def _normalize_moments(v, target_mean, target_second_moment, passes=2):
    """Rescale v to exact target mean / second moment, via the reference's
    tau/lambda formula, applied twice (as in the reference)."""
    v = v.copy()
    for _ in range(passes):
        v_prime = np.mean(v)
        E_prime = np.mean(v * v)
        tau = np.sqrt((E_prime - 0.5 * v_prime**2) / (target_second_moment - 0.5 * target_mean**2))
        lam = v_prime - tau * target_mean
        v = (v - lam) / tau
    return v


def _init_one_species_two_stream(N, alpha, density, temp, mass, drift, u_v2, Lx, k, rng):
    """One species' positions/velocities/weights for the (possibly split)
    two-stream initial condition. drift=0 gives an ordinary Maxwellian
    (both halves become statistically identical)."""
    x = _stratified_positions(N, alpha, Lx, k, rng)

    v1 = _normalize_moments(rng.standard_normal(N), target_mean=0.0, target_second_moment=1.0)
    v2 = _normalize_moments(rng.standard_normal(N), target_mean=0.0, target_second_moment=1.0)

    T = np.sqrt(temp / mass)  # == 1 for the reference's temp=mass=1
    half = N // 2
    v1_full = np.empty(N, dtype=float)
    v1_full[:half] = drift + T * v1[:half]
    v1_full[half:] = -drift + T * v1[half:]
    v2_full = u_v2 + T * v2

    # Constant weight per particle (variable particle count per cell
    # instead), matching the reference's un-weighted representation. Total
    # weight sums to density*Lx exactly, for this species alone.
    w = np.full(N, density * Lx / N, dtype=float)
    return x, v1_full, v2_full, w


def initialize_particles_two_stream(par):
    """
    Two-stream instability, 1D-2V, Vlasov-Ampere, general two-species
    version. See module docstring for the full algorithm and field
    mapping. Set u1_v1_0/u2_v1_0 to 0 for a species you want as an
    ordinary (optionally alpha_s-perturbed) Maxwellian rather than a split
    stream -- e.g. u2_v1_0=0 (and alpha2=0 for a spatially uniform
    background too) for a plain ion background, as in linear Landau
    damping. Species 1 and species 2 are otherwise fully independent:
    different mass/charge/temperature/density are expected and supported.
    """
    rng = np.random.default_rng(par.random_seed)

    x1, v11, v12, w1 = _init_one_species_two_stream(
        par.N1, par.alpha1, par.n1, par.temp1, par.m1,
        par.u1_v1_0, par.u1_v2_0, par.Lx, par.k, rng,
    )
    x2, v21, v22, w2 = _init_one_species_two_stream(
        par.N2, par.alpha2, par.n2, par.temp2, par.m2,
        par.u2_v1_0, par.u2_v2_0, par.Lx, par.k, rng,
    )
    return x1, v11, v12, w1, x2, v21, v22, w2