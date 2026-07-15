"""
Numba implementation of the two-species inhomogeneous C-PIC collision force
for the 1D-2V setting.

For each species, Steps I-II use a separate phase-space cell list in (x,v1,v2),
with its own velocity scales eps_s_v1, eps_s_v2.  Step III uses source-species
cell lists in x, optionally split by random batch.

This is a drop-in replacement for the original collisions_numba.py: the two
public entry points (compute_collision_acceleration and
compute_regularized_density_at_particles) have identical signatures and
return values, and implement the *same method* (same regularised functional,
same cell list / random batch scheme -- see Bailo, Carrillo & Hu (2024),
Sec. 2.2.3, 2.3). What changed is purely implementation:

  1. Cell membership is now resolved via a physically cell-sorted, contiguous
     array (Functions/cell_sort.py) instead of a pointer-chasing linked list
     (head/nxt). The set of particles found in each cell, and therefore every
     sum computed over them, is unchanged; only the memory-access pattern
     used to find them is different. This is the single biggest lever here:
     Step III's O(Nx log(Nx) (Nv^2 Nc)^2 / R) pairwise sum (the dominant cost
     per the reference papers) is entirely bound by how fast the neighbour
     particles' data can be streamed from memory.
  2. Step III's self-species and cross-species sweeps (previously 2 separate
     kernel launches per receiver species, 4 total) are fused into one pass
     per receiver species (2 total), reusing the receiver-side load (position,
     velocity, F/m, cell membership) for both source terms instead of loading
     it twice.
  3. For gam=-2 (the 2V Coulomb case this codebase's tests use), r2**(gam/2)
     is replaced by a plain reciprocal 1/r2, avoiding a pow()/exp(log()) call
     in the innermost loop of the dominant term. Arbitrary gam still works
     via the original pow()-based formula.
  4. Loop-invariant reciprocals (1/eta, 1/eps1, 1/eps2, 1/(eps1*eps2), ...)
     are hoisted out of the inner pairwise loops instead of being
     recomputed (as a division) on every (p, q) pair.
  5. Each particle's own cell-index components (its spatial/velocity cell,
     or spatial/batch cell) are computed once and reused across the passes
     that need them, instead of being recomputed by floor/clip arithmetic
     in each of Steps I, II, III separately.

All five changes were validated against the original linked-list
implementation on many randomised small cases (matching to floating-point
roundoff, i.e. differences at the 1e-15..1e-16 level from summation-order
changes only) before being adopted here; see test_regression.py.
"""
import numpy as np
from numba import njit, prange

from .splines_numba import spline_value_derivative, periodic_delta
from .cell_sort import counting_sort_by_cell, default_nchunks


# ------------------------------------------------------------ spline-order-3 fast path
@njit(inline="always", fastmath=True)
def _cubic_value(u):
    a = u if u >= 0.0 else -u
    if a <= 1.0:
        return (4.0 - 6.0 * u * u + 3.0 * a * a * a) / 6.0
    if a <= 2.0:
        t = 2.0 - a
        return t * t * t / 6.0
    return 0.0


@njit(inline="always", fastmath=True)
def _cubic_value_derivative(u):
    a = u if u >= 0.0 else -u
    if a <= 1.0:
        return (4.0 - 6.0 * u * u + 3.0 * a * a * a) / 6.0, (-12.0 * u + 9.0 * u * a) / 6.0
    if a <= 2.0:
        t = 2.0 - a
        if u > 0.0:
            return t * t * t / 6.0, -0.5 * t * t
        if u < 0.0:
            return t * t * t / 6.0, 0.5 * t * t
        return t * t * t / 6.0, 0.0
    return 0.0, 0.0


# ------------------------------------------------------------ cell components
@njit(parallel=True, cache=True)
def _build_phase_cell_components(x, v1, v2, eta, eps1, eps2, Nx, radius):
    """Per-particle (x,v1,v2) cell membership for Steps I-II, computed once
    and reused by both passes (the original recomputed this from scratch in
    each of Step I and Step II separately, on top of a third time while
    building the linked list)."""
    N = x.shape[0]
    v1_min = np.min(v1)
    v1_max = np.max(v1)
    v2_min = np.min(v2)
    v2_max = np.max(v2)
    v1_origin = v1_min - (radius + 1.0) * eps1
    v2_origin = v2_min - (radius + 1.0) * eps2
    ncv1 = int(np.floor((v1_max - v1_origin + (radius + 1.0) * eps1) / eps1)) + 2
    ncv2 = int(np.floor((v2_max - v2_origin + (radius + 1.0) * eps2) / eps2)) + 2
    if ncv1 < 1:
        ncv1 = 1
    if ncv2 < 1:
        ncv2 = 1

    ix0 = np.empty(N, dtype=np.int64)
    iv10 = np.empty(N, dtype=np.int64)
    iv20 = np.empty(N, dtype=np.int64)
    cell_of = np.empty(N, dtype=np.int64)

    for p in prange(N):
        ix = int(np.floor(x[p] / eta))
        if ix < 0 or ix >= Nx:
            ix = ix % Nx
        iv1 = int(np.floor((v1[p] - v1_origin) / eps1))
        iv2 = int(np.floor((v2[p] - v2_origin) / eps2))
        if iv1 < 0:
            iv1 = 0
        elif iv1 >= ncv1:
            iv1 = ncv1 - 1
        if iv2 < 0:
            iv2 = 0
        elif iv2 >= ncv2:
            iv2 = ncv2 - 1
        ix0[p] = ix
        iv10[p] = iv1
        iv20[p] = iv2
        cell_of[p] = (ix * ncv1 + iv1) * ncv2 + iv2

    return ix0, iv10, iv20, cell_of, v1_origin, v2_origin, ncv1, ncv2


@njit(parallel=True, cache=True)
def _build_xbatch_components(x, batch_ids, eta, Nx, R_eff):
    """Per-particle (x, batch) cell membership for Step III."""
    N = x.shape[0]
    ix0 = np.empty(N, dtype=np.int64)
    b_arr = np.empty(N, dtype=np.int64)
    cell_of = np.empty(N, dtype=np.int64)
    for p in prange(N):
        ix = int(np.floor(x[p] / eta))
        if ix < 0 or ix >= Nx:
            ix = ix % Nx
        b = batch_ids[p]
        if b < 0:
            b = 0
        elif b >= R_eff:
            b = R_eff - 1
        ix0[p] = ix
        b_arr[p] = b
        cell_of[p] = ix * R_eff + b
    return ix0, b_arr, cell_of


# ------------------------------------------------------------ Steps I & II (sorted)
@njit(parallel=True, fastmath=True, cache=True)
def _compute_ft_grad_pass_sorted(x_s, v1_s, v2_s, w_s, ix0_s, iv10_s, iv20_s, cell_start,
                                  Lx, eta, eps1, eps2, Nx, order, radius, ncv1, ncv2):
    N = x_s.shape[0]
    ft = np.empty(N, dtype=np.float64)
    g1 = np.empty(N, dtype=np.float64)
    g2 = np.empty(N, dtype=np.float64)
    rx = int(np.ceil(radius))
    rv1 = int(np.ceil(radius))
    rv2 = int(np.ceil(radius))

    inv_eta = 1.0 / eta
    inv_eps1 = 1.0 / eps1
    inv_eps2 = 1.0 / eps2
    inv_eps1_eps2 = 1.0 / (eps1 * eps2)
    inv_eps1sq_eps2 = 1.0 / (eps1 * eps1 * eps2)
    inv_eps1_eps2sq = 1.0 / (eps1 * eps2 * eps2)

    for p in prange(N):
        ix0 = ix0_s[p]
        iv10 = iv10_s[p]
        iv20 = iv20_s[p]
        xp = x_s[p]
        v1p = v1_s[p]
        v2p = v2_s[p]

        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for dix in range(-rx, rx + 1):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            for di1 in range(-rv1, rv1 + 1):
                iv1 = iv10 + di1
                if iv1 < 0 or iv1 >= ncv1:
                    continue
                for di2 in range(-rv2, rv2 + 1):
                    iv2 = iv20 + di2
                    if iv2 < 0 or iv2 >= ncv2:
                        continue
                    cell = (ix * ncv1 + iv1) * ncv2 + iv2
                    start = cell_start[cell]
                    end = cell_start[cell + 1]
                    for qi in range(start, end):
                        dx = periodic_delta(xp, x_s[qi], Lx)
                        bx, _ = spline_value_derivative(dx * inv_eta, order)
                        if bx != 0.0:
                            dv1 = v1p - v1_s[qi]
                            dv2 = v2p - v2_s[qi]
                            b1, db1 = spline_value_derivative(dv1 * inv_eps1, order)
                            if b1 != 0.0 or db1 != 0.0:
                                b2, db2 = spline_value_derivative(dv2 * inv_eps2, order)
                                if b2 != 0.0 or db2 != 0.0:
                                    wx = bx * inv_eta
                                    phi = b1 * b2 * inv_eps1_eps2
                                    grad1 = db1 * b2 * inv_eps1sq_eps2
                                    grad2 = b1 * db2 * inv_eps1_eps2sq
                                    wq = w_s[qi]
                                    s0 += wq * wx * phi
                                    s1 += wq * wx * grad1
                                    s2 += wq * wx * grad2
        ft[p] = s0
        g1[p] = s1
        g2[p] = s2
    return ft, g1, g2


@njit(parallel=True, fastmath=True, cache=True)
def _compute_F_second_pass_sorted(x_s, v1_s, v2_s, w_s, ft_s, grad1_s, grad2_s,
                                   ix0_s, iv10_s, iv20_s, cell_start,
                                   Lx, eta, eps1, eps2, Nx, order, radius, ncv1, ncv2):
    N = x_s.shape[0]
    F1 = np.empty(N, dtype=np.float64)
    F2 = np.empty(N, dtype=np.float64)
    rx = int(np.ceil(radius))
    rv1 = int(np.ceil(radius))
    rv2 = int(np.ceil(radius))

    inv_eta = 1.0 / eta
    inv_eps1 = 1.0 / eps1
    inv_eps2 = 1.0 / eps2
    inv_eps1sq_eps2 = 1.0 / (eps1 * eps1 * eps2)
    inv_eps1_eps2sq = 1.0 / (eps1 * eps2 * eps2)

    for p in prange(N):
        ix0 = ix0_s[p]
        iv10 = iv10_s[p]
        iv20 = iv20_s[p]
        xp = x_s[p]
        v1p = v1_s[p]
        v2p = v2_s[p]

        s1 = 0.0
        s2 = 0.0
        for dix in range(-rx, rx + 1):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            for di1 in range(-rv1, rv1 + 1):
                iv1 = iv10 + di1
                if iv1 < 0 or iv1 >= ncv1:
                    continue
                for di2 in range(-rv2, rv2 + 1):
                    iv2 = iv20 + di2
                    if iv2 < 0 or iv2 >= ncv2:
                        continue
                    cell = (ix * ncv1 + iv1) * ncv2 + iv2
                    start = cell_start[cell]
                    end = cell_start[cell + 1]
                    for qi in range(start, end):
                        dx = periodic_delta(xp, x_s[qi], Lx)
                        bx, _ = spline_value_derivative(dx * inv_eta, order)
                        if bx != 0.0:
                            dv1 = v1p - v1_s[qi]
                            dv2 = v2p - v2_s[qi]
                            b1, db1 = spline_value_derivative(dv1 * inv_eps1, order)
                            if b1 != 0.0 or db1 != 0.0:
                                b2, db2 = spline_value_derivative(dv2 * inv_eps2, order)
                                if b2 != 0.0 or db2 != 0.0:
                                    wx = bx * inv_eta
                                    gradv1 = db1 * b2 * inv_eps1sq_eps2
                                    gradv2 = b1 * db2 * inv_eps1_eps2sq
                                    coef = w_s[qi] / ft_s[qi]
                                    s1 += coef * wx * gradv1
                                    s2 += coef * wx * gradv2
        F1[p] = grad1_s[p] / ft_s[p] + s1
        F2[p] = grad2_s[p] / ft_s[p] + s2
    return F1, F2


@njit(parallel=True, fastmath=True, cache=True)
def _compute_ft_grad_pass_sorted_cubic(x_s, v1_s, v2_s, w_s, ix0_s, iv10_s, iv20_s, cell_start,
                                       Lx, eta, eps1, eps2, Nx, ncv1, ncv2):
    N = x_s.shape[0]
    ft = np.empty(N, dtype=np.float64)
    g1 = np.empty(N, dtype=np.float64)
    g2 = np.empty(N, dtype=np.float64)

    inv_eta = 1.0 / eta
    inv_eps1 = 1.0 / eps1
    inv_eps2 = 1.0 / eps2
    inv_eps1_eps2 = 1.0 / (eps1 * eps2)
    inv_eps1sq_eps2 = 1.0 / (eps1 * eps1 * eps2)
    inv_eps1_eps2sq = 1.0 / (eps1 * eps2 * eps2)

    for p in prange(N):
        ix0 = ix0_s[p]
        iv10 = iv10_s[p]
        iv20 = iv20_s[p]
        xp = x_s[p]
        v1p = v1_s[p]
        v2p = v2_s[p]

        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for dix in range(-2, 3):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            for di1 in range(-2, 3):
                iv1 = iv10 + di1
                if iv1 < 0 or iv1 >= ncv1:
                    continue
                for di2 in range(-2, 3):
                    iv2 = iv20 + di2
                    if iv2 < 0 or iv2 >= ncv2:
                        continue
                    cell = (ix * ncv1 + iv1) * ncv2 + iv2
                    start = cell_start[cell]
                    end = cell_start[cell + 1]
                    for qi in range(start, end):
                        bx = _cubic_value(periodic_delta(xp, x_s[qi], Lx) * inv_eta)
                        if bx != 0.0:
                            dv1 = v1p - v1_s[qi]
                            dv2 = v2p - v2_s[qi]
                            b1, db1 = _cubic_value_derivative(dv1 * inv_eps1)
                            if b1 != 0.0:
                                b2, db2 = _cubic_value_derivative(dv2 * inv_eps2)
                                if b2 != 0.0:
                                    wx = bx * inv_eta
                                    phi = b1 * b2 * inv_eps1_eps2
                                    grad1 = db1 * b2 * inv_eps1sq_eps2
                                    grad2 = b1 * db2 * inv_eps1_eps2sq
                                    wq = w_s[qi]
                                    s0 += wq * wx * phi
                                    s1 += wq * wx * grad1
                                    s2 += wq * wx * grad2
        ft[p] = s0
        g1[p] = s1
        g2[p] = s2
    return ft, g1, g2


@njit(parallel=True, fastmath=True, cache=True)
def _compute_F_second_pass_sorted_cubic(x_s, v1_s, v2_s, w_s, ft_s, grad1_s, grad2_s,
                                        ix0_s, iv10_s, iv20_s, cell_start,
                                        Lx, eta, eps1, eps2, Nx, ncv1, ncv2):
    N = x_s.shape[0]
    F1 = np.empty(N, dtype=np.float64)
    F2 = np.empty(N, dtype=np.float64)

    inv_eta = 1.0 / eta
    inv_eps1 = 1.0 / eps1
    inv_eps2 = 1.0 / eps2
    inv_eps1sq_eps2 = 1.0 / (eps1 * eps1 * eps2)
    inv_eps1_eps2sq = 1.0 / (eps1 * eps2 * eps2)

    for p in prange(N):
        ix0 = ix0_s[p]
        iv10 = iv10_s[p]
        iv20 = iv20_s[p]
        xp = x_s[p]
        v1p = v1_s[p]
        v2p = v2_s[p]

        s1 = 0.0
        s2 = 0.0
        for dix in range(-2, 3):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            for di1 in range(-2, 3):
                iv1 = iv10 + di1
                if iv1 < 0 or iv1 >= ncv1:
                    continue
                for di2 in range(-2, 3):
                    iv2 = iv20 + di2
                    if iv2 < 0 or iv2 >= ncv2:
                        continue
                    cell = (ix * ncv1 + iv1) * ncv2 + iv2
                    start = cell_start[cell]
                    end = cell_start[cell + 1]
                    for qi in range(start, end):
                        bx = _cubic_value(periodic_delta(xp, x_s[qi], Lx) * inv_eta)
                        if bx != 0.0:
                            dv1 = v1p - v1_s[qi]
                            dv2 = v2p - v2_s[qi]
                            b1, db1 = _cubic_value_derivative(dv1 * inv_eps1)
                            if b1 != 0.0:
                                b2, db2 = _cubic_value_derivative(dv2 * inv_eps2)
                                if b2 != 0.0:
                                    wx = bx * inv_eta
                                    gradv1 = db1 * b2 * inv_eps1sq_eps2
                                    gradv2 = b1 * db2 * inv_eps1_eps2sq
                                    coef = w_s[qi] / ft_s[qi]
                                    s1 += coef * wx * gradv1
                                    s2 += coef * wx * gradv2
        F1[p] = grad1_s[p] / ft_s[p] + s1
        F2[p] = grad2_s[p] / ft_s[p] + s2
    return F1, F2


def _compute_species_F(x, v1, v2, w, eps1, eps2, par):
    radius = par.spline_radius
    ix0, iv10, iv20, cell_of, v1_origin, v2_origin, ncv1, ncv2 = _build_phase_cell_components(
        x, v1, v2, par.eta, eps1, eps2, par.Nx, radius
    )
    ncells = par.Nx * ncv1 * ncv2
    nchunks = default_nchunks(x.shape[0], par.sort_oversample)
    sort_idx, cell_start = counting_sort_by_cell(cell_of, ncells, nchunks)

    x_s = x[sort_idx]
    v1_s = v1[sort_idx]
    v2_s = v2[sort_idx]
    w_s = w[sort_idx]
    ix0_s = ix0[sort_idx]
    iv10_s = iv10[sort_idx]
    iv20_s = iv20[sort_idx]

    if par.spline_order == 3:
        ft_s, g1_s, g2_s = _compute_ft_grad_pass_sorted_cubic(
            x_s, v1_s, v2_s, w_s, ix0_s, iv10_s, iv20_s, cell_start,
            par.Lx, par.eta, eps1, eps2, par.Nx, ncv1, ncv2
        )
        F1_s, F2_s = _compute_F_second_pass_sorted_cubic(
            x_s, v1_s, v2_s, w_s, ft_s, g1_s, g2_s, ix0_s, iv10_s, iv20_s, cell_start,
            par.Lx, par.eta, eps1, eps2, par.Nx, ncv1, ncv2
        )
    else:
        ft_s, g1_s, g2_s = _compute_ft_grad_pass_sorted(
            x_s, v1_s, v2_s, w_s, ix0_s, iv10_s, iv20_s, cell_start,
            par.Lx, par.eta, eps1, eps2, par.Nx, par.spline_order, radius, ncv1, ncv2
        )
        F1_s, F2_s = _compute_F_second_pass_sorted(
            x_s, v1_s, v2_s, w_s, ft_s, g1_s, g2_s, ix0_s, iv10_s, iv20_s, cell_start,
            par.Lx, par.eta, eps1, eps2, par.Nx, par.spline_order, radius, ncv1, ncv2
        )

    N = x.shape[0]
    ft = np.empty(N, dtype=np.float64)
    F1 = np.empty(N, dtype=np.float64)
    F2 = np.empty(N, dtype=np.float64)
    ft[sort_idx] = ft_s
    F1[sort_idx] = F1_s
    F2[sort_idx] = F2_s
    return ft, F1, F2


# ------------------------------------------------------------ Step III (sorted, fused)
@njit(parallel=True, fastmath=True, cache=True)
def _compute_pair_accel_fused_sorted(
        xr_s, v1r_s, v2r_s, Fr1m_s, Fr2m_s, ixr_s, br_s, cell_start_self, w_self_s,
        xc_s, v1c_s, v2c_s, wc_s, Fc1m_s, Fc2m_s, cell_start_cross,
        Lx, eta, Nx, order, radius, R_eff,
        scale_self, scale_cross, half_gam, B_self, B_cross, use_reciprocal):
    """Step III for one receiver species, fused: both the self-species
    (B_self) and cross-species (B_cross) source sweeps are accumulated in
    the same pass over the receiver particles, instead of two separate
    kernel launches that each reload the receiver's position/velocity/F
    (this replaces what were previously two independent calls to the
    equivalent of _compute_pair_accel_xbatch)."""
    Nr = xr_s.shape[0]
    a1 = np.empty(Nr, dtype=np.float64)
    a2 = np.empty(Nr, dtype=np.float64)
    rx = int(np.ceil(radius))
    inv_eta = 1.0 / eta

    for p in prange(Nr):
        ix0 = ixr_s[p]
        b = br_s[p]
        xp = xr_s[p]
        v1p = v1r_s[p]
        v2p = v2r_s[p]
        F1p = Fr1m_s[p]
        F2p = Fr2m_s[p]
        s1 = 0.0
        s2 = 0.0

        for dix in range(-rx, rx + 1):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            cell = ix * R_eff + b

            # ---- self-species source ----
            start = cell_start_self[cell]
            end = cell_start_self[cell + 1]
            for qi in range(start, end):
                if qi != p:
                    dx = periodic_delta(xp, xr_s[qi], Lx)
                    bx, _ = spline_value_derivative(dx * inv_eta, order)
                    if bx != 0.0:
                        dv1 = v1p - v1r_s[qi]
                        dv2 = v2p - v2r_s[qi]
                        r2 = dv1 * dv1 + dv2 * dv2
                        if r2 > 0.0:
                            if use_reciprocal:
                                coeff = B_self / r2
                            else:
                                coeff = B_self * (r2 ** half_gam)
                            A11 = coeff * dv2 * dv2
                            A12 = -coeff * dv1 * dv2
                            A22 = coeff * dv1 * dv1
                            bF1 = Fr1m_s[qi] - F1p
                            bF2 = Fr2m_s[qi] - F2p
                            psix = bx * inv_eta
                            wq = w_self_s[qi] * psix * scale_self
                            s1 += wq * (A11 * bF1 + A12 * bF2)
                            s2 += wq * (A12 * bF1 + A22 * bF2)

            # ---- cross-species source ----
            start = cell_start_cross[cell]
            end = cell_start_cross[cell + 1]
            for qi in range(start, end):
                dx = periodic_delta(xp, xc_s[qi], Lx)
                bx, _ = spline_value_derivative(dx * inv_eta, order)
                if bx != 0.0:
                    dv1 = v1p - v1c_s[qi]
                    dv2 = v2p - v2c_s[qi]
                    r2 = dv1 * dv1 + dv2 * dv2
                    if r2 > 0.0:
                        if use_reciprocal:
                            coeff = B_cross / r2
                        else:
                            coeff = B_cross * (r2 ** half_gam)
                        A11 = coeff * dv2 * dv2
                        A12 = -coeff * dv1 * dv2
                        A22 = coeff * dv1 * dv1
                        bF1 = Fc1m_s[qi] - F1p
                        bF2 = Fc2m_s[qi] - F2p
                        psix = bx * inv_eta
                        wq = wc_s[qi] * psix * scale_cross
                        s1 += wq * (A11 * bF1 + A12 * bF2)
                        s2 += wq * (A12 * bF1 + A22 * bF2)
        a1[p] = s1
        a2[p] = s2
    return a1, a2


@njit(parallel=True, fastmath=True, cache=True)
def _compute_pair_accel_fused_sorted_cubic_gamm2(
        xr_s, v1r_s, v2r_s, Fr1m_s, Fr2m_s, ixr_s, br_s, cell_start_self, w_self_s,
        xc_s, v1c_s, v2c_s, wc_s, Fc1m_s, Fc2m_s, cell_start_cross,
        Lx, eta, Nx, R_eff, scale_self, scale_cross, B_self, B_cross):
    Nr = xr_s.shape[0]
    a1 = np.empty(Nr, dtype=np.float64)
    a2 = np.empty(Nr, dtype=np.float64)
    inv_eta = 1.0 / eta

    for p in prange(Nr):
        ix0 = ixr_s[p]
        b = br_s[p]
        xp = xr_s[p]
        v1p = v1r_s[p]
        v2p = v2r_s[p]
        F1p = Fr1m_s[p]
        F2p = Fr2m_s[p]
        s1 = 0.0
        s2 = 0.0

        for dix in range(-2, 3):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            cell = ix * R_eff + b

            start = cell_start_self[cell]
            end = cell_start_self[cell + 1]
            for qi in range(start, end):
                if qi != p:
                    bx = _cubic_value(periodic_delta(xp, xr_s[qi], Lx) * inv_eta)
                    if bx != 0.0:
                        dv1 = v1p - v1r_s[qi]
                        dv2 = v2p - v2r_s[qi]
                        r2 = dv1 * dv1 + dv2 * dv2
                        if r2 > 0.0:
                            coeff = B_self / r2
                            A11 = coeff * dv2 * dv2
                            A12 = -coeff * dv1 * dv2
                            A22 = coeff * dv1 * dv1
                            bF1 = Fr1m_s[qi] - F1p
                            bF2 = Fr2m_s[qi] - F2p
                            wq = w_self_s[qi] * bx * inv_eta * scale_self
                            s1 += wq * (A11 * bF1 + A12 * bF2)
                            s2 += wq * (A12 * bF1 + A22 * bF2)

            start = cell_start_cross[cell]
            end = cell_start_cross[cell + 1]
            for qi in range(start, end):
                bx = _cubic_value(periodic_delta(xp, xc_s[qi], Lx) * inv_eta)
                if bx != 0.0:
                    dv1 = v1p - v1c_s[qi]
                    dv2 = v2p - v2c_s[qi]
                    r2 = dv1 * dv1 + dv2 * dv2
                    if r2 > 0.0:
                        coeff = B_cross / r2
                        A11 = coeff * dv2 * dv2
                        A12 = -coeff * dv1 * dv2
                        A22 = coeff * dv1 * dv1
                        bF1 = Fc1m_s[qi] - F1p
                        bF2 = Fc2m_s[qi] - F2p
                        wq = wc_s[qi] * bx * inv_eta * scale_cross
                        s1 += wq * (A11 * bF1 + A12 * bF2)
                        s2 += wq * (A12 * bF1 + A22 * bF2)
        a1[p] = s1
        a2[p] = s2
    return a1, a2


def compute_collision_acceleration(x1, v11, v12, w1, x2, v21, v22, w2,
                                   batch1, batch2, par):
    """
    Return collisional accelerations for both species plus regularized densities.

    Returns:
        a11,a12,a21,a22, ft1,ft2, F11,F12,F21,F22
    where a11/a12 are species-1 velocity accelerations and a21/a22 species-2.
    """
    if not par.collision_active:
        z1 = np.zeros_like(v11)
        z2 = np.zeros_like(v21)
        return z1, z1.copy(), z2, z2.copy(), None, None, None, None, None, None

    ft1, F11, F12 = _compute_species_F(x1, v11, v12, w1, par.eps1_v1, par.eps1_v2, par)
    ft2, F21, F22 = _compute_species_F(x2, v21, v22, w2, par.eps2_v1, par.eps2_v2, par)

    F11m = F11 / par.m1
    F12m = F12 / par.m1
    F21m = F21 / par.m2
    F22m = F22 / par.m2

    if par.random_batch and par.random_batches > 1:
        R_eff = par.random_batches
        scale_same_1 = R_eff * (x1.shape[0] - 1.0) / (x1.shape[0] - R_eff)
        scale_same_2 = R_eff * (x2.shape[0] - 1.0) / (x2.shape[0] - R_eff)
        scale_cross = float(R_eff)
    else:
        R_eff = 1
        scale_same_1 = 1.0
        scale_same_2 = 1.0
        scale_cross = 1.0

    ix0_1, b1_arr, cell_of_1 = _build_xbatch_components(x1, batch1, par.eta, par.Nx, R_eff)
    ix0_2, b2_arr, cell_of_2 = _build_xbatch_components(x2, batch2, par.eta, par.Nx, R_eff)
    ncells_xb = par.Nx * R_eff
    nchunks1 = default_nchunks(x1.shape[0], par.sort_oversample)
    nchunks2 = default_nchunks(x2.shape[0], par.sort_oversample)
    sort_idx1, cell_start1 = counting_sort_by_cell(cell_of_1, ncells_xb, nchunks1)
    sort_idx2, cell_start2 = counting_sort_by_cell(cell_of_2, ncells_xb, nchunks2)

    x1_s = x1[sort_idx1]; v11_s = v11[sort_idx1]; v12_s = v12[sort_idx1]; w1_s = w1[sort_idx1]
    F11m_s = F11m[sort_idx1]; F12m_s = F12m[sort_idx1]
    ix0_1_s = ix0_1[sort_idx1]; b1_s = b1_arr[sort_idx1]

    x2_s = x2[sort_idx2]; v21_s = v21[sort_idx2]; v22_s = v22[sort_idx2]; w2_s = w2[sort_idx2]
    F21m_s = F21m[sort_idx2]; F22m_s = F22m[sort_idx2]
    ix0_2_s = ix0_2[sort_idx2]; b2_s = b2_arr[sort_idx2]

    half_gam = 0.5 * par.gam
    use_recip = (par.gam == -2.0)
    C = par.collision_strength

    if par.spline_order == 3 and par.gam == -2.0:
        a1_sorted, a2_sorted = _compute_pair_accel_fused_sorted_cubic_gamm2(
            x1_s, v11_s, v12_s, F11m_s, F12m_s, ix0_1_s, b1_s, cell_start1, w1_s,
            x2_s, v21_s, v22_s, w2_s, F21m_s, F22m_s, cell_start2,
            par.Lx, par.eta, par.Nx, R_eff, scale_same_1, scale_cross, C * par.B11, C * par.B21
        )
        a1_sorted2, a2_sorted2 = _compute_pair_accel_fused_sorted_cubic_gamm2(
            x2_s, v21_s, v22_s, F21m_s, F22m_s, ix0_2_s, b2_s, cell_start2, w2_s,
            x1_s, v11_s, v12_s, w1_s, F11m_s, F12m_s, cell_start1,
            par.Lx, par.eta, par.Nx, R_eff, scale_same_2, scale_cross, C * par.B22, C * par.B12
        )
    else:
        a1_sorted, a2_sorted = _compute_pair_accel_fused_sorted(
            x1_s, v11_s, v12_s, F11m_s, F12m_s, ix0_1_s, b1_s, cell_start1, w1_s,
            x2_s, v21_s, v22_s, w2_s, F21m_s, F22m_s, cell_start2,
            par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, R_eff,
            scale_same_1, scale_cross, half_gam, C * par.B11, C * par.B21, use_recip
        )
        a1_sorted2, a2_sorted2 = _compute_pair_accel_fused_sorted(
            x2_s, v21_s, v22_s, F21m_s, F22m_s, ix0_2_s, b2_s, cell_start2, w2_s,
            x1_s, v11_s, v12_s, w1_s, F11m_s, F12m_s, cell_start1,
            par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, R_eff,
            scale_same_2, scale_cross, half_gam, C * par.B22, C * par.B12, use_recip
        )

    a11 = np.empty_like(v11)
    a12 = np.empty_like(v12)
    a11[sort_idx1] = a1_sorted
    a12[sort_idx1] = a2_sorted

    a21 = np.empty_like(v21)
    a22 = np.empty_like(v22)
    a21[sort_idx2] = a1_sorted2
    a22[sort_idx2] = a2_sorted2

    return (
        a11, a12, a21, a22,
        ft1, ft2, F11, F12, F21, F22,
    )


def compute_regularized_density_at_particles(x1, v11, v12, w1, x2, v21, v22, w2, par):
    """Compute ftilde_i at particles for diagnostics."""
    ft1, _, _ = _compute_species_F(x1, v11, v12, w1, par.eps1_v1, par.eps1_v2, par)
    ft2, _, _ = _compute_species_F(x2, v21, v22, w2, par.eps2_v1, par.eps2_v2, par)
    return ft1, ft2
