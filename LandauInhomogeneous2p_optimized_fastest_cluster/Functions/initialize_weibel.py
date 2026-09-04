"""
Particle initialisation for the two-species Weibel instability, 1D-2V.

Single-species reference (Bailo, Carrillo & Hu 2024, Sec 3.2.2, borrowed
from Cheng, Christlieb & Zhong 2014):

    f0(x, v1, v2) = (1/(pi*beta)) * [ exp(-v1^2/beta) * exp(-(v2-c)^2/beta)
                                     + exp(-v1^2/beta) * exp(-(v2+c)^2/beta) ]

i.e. two counter-streaming Maxwellian beams split in v2 (not v1, unlike the
two-stream test), uniform in x, with c=0.3 and beta=1e-2. The instability
is seeded purely through the initial magnetic field, B3(0,x) = alpha*sin(kx)
-- see Functions/fields.py's initialize_fields(initial_field="analytic_weibel")
-- rather than through a density perturbation, so alpha1=alpha2=0 here.

This module extends that single-species datum to two mobile species the
same way Functions/initialize_two_stream.py extends the two-stream datum:
species 1 carries the interesting physics (the two beams, split in v2),
and species 2 is, by default, an ordinary (u2_v2_0=0) Maxwellian
background providing charge neutrality -- exactly analogous to alpha2=0
in initialize.py and u2_v1_0=0 in initialize_two_stream.py.

Both species reuse par.temp_s/par.m_s for the common thermal variance of
v1 and v2 (== beta/2 in the notation above), and par.u_s_v2_0 as the beam
half-separation c (par.u_s_v1_0 is the mean of the un-split v1 component,
0 in the reference datum). Set u1_v2_0=0 to recover an ordinary Maxwellian
for species 1 too, exactly as u1_v1_0=0/u2_v1_0=0 do in
initialize_two_stream.py.
"""
import numpy as np

from .initialize_two_stream import _stratified_positions, _normalize_moments


def _init_one_species_weibel(N, density, temp, mass, drift, u_v1, Lx, k, rng):
    """One species' positions/velocities/weights for the (possibly split)
    Weibel initial condition. drift=0 gives an ordinary Maxwellian in v2
    (both halves become statistically identical), matching the
    u_*_v1_0=0 convention in initialize_two_stream.py.

    Positions are uniform in x (alpha=0): the Weibel datum has no spatial
    density perturbation, unlike Landau damping/two-stream.
    """
    x = _stratified_positions(N, 0.0, Lx, k, rng)

    v1 = _normalize_moments(rng.standard_normal(N), target_mean=0.0, target_second_moment=1.0)
    v2 = _normalize_moments(rng.standard_normal(N), target_mean=0.0, target_second_moment=1.0)

    T = np.sqrt(temp / mass)  # thermal std of both v1 and v2; == sqrt(beta/2) for the reference datum
    v1_full = u_v1 + T * v1

    half = N // 2
    v2_full = np.empty(N, dtype=float)
    v2_full[:half] = drift + T * v2[:half]
    v2_full[half:] = -drift + T * v2[half:]

    # Constant weight per particle, matching the reference's un-weighted
    # representation (see initialize_two_stream.py). Total weight sums to
    # density*Lx exactly, for this species alone.
    w = np.full(N, density * Lx / N, dtype=float)
    return x, v1_full, v2_full, w


def initialize_particles_weibel_2species(par):
    """
    Weibel instability, 1D-2V, Vlasov-Maxwell-Landau, two-species version.
    See module docstring for the full algorithm and species mapping.
    Species 1 carries the two counter-streaming beams in v2 by default
    (u1_v2_0=0.3); species 2 defaults to a plain Maxwellian background
    (u2_v2_0=0.0). Species 1 and species 2 are otherwise fully
    independent: different mass/charge/temperature/density are expected
    and supported, exactly as in initialize_two_stream.py.
    """
    rng = np.random.default_rng(par.random_seed)

    x1, v11, v12, w1 = _init_one_species_weibel(
        par.N1, par.n1, par.temp1, par.m1,
        par.u1_v2_0, par.u1_v1_0, par.Lx, par.k, rng,
    )
    x2, v21, v22, w2 = _init_one_species_weibel(
        par.N2, par.n2, par.temp2, par.m2,
        par.u2_v2_0, par.u2_v1_0, par.Lx, par.k, rng,
    )
    return x1, v11, v12, w1, x2, v21, v22, w2
