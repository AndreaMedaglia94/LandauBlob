"""
Numba implementation of the two-species inhomogeneous C-PIC collision force
for the 1D-2V setting.

For each species, Steps I-II use a separate phase-space cell list in (x,v1,v2),
with its own velocity scales eps_s_v1, eps_s_v2.  Step III uses source-species
cell lists in x, optionally split by random batch.
"""
import numpy as np
from numba import njit, prange

from .splines_numba import spline_value_derivative, periodic_delta


@njit(cache=True)
def _minmax(a):
    mn = a[0]
    mx = a[0]
    for i in range(1, a.shape[0]):
        val = a[i]
        if val < mn:
            mn = val
        if val > mx:
            mx = val
    return mn, mx


@njit(cache=True)
def _build_phase_cell_list(x, v1, v2, Lx, eta, eps1, eps2, Nx, radius):
    N = x.shape[0]
    v1_min, v1_max = _minmax(v1)
    v2_min, v2_max = _minmax(v2)

    v1_origin = v1_min - (radius + 1.0) * eps1
    v2_origin = v2_min - (radius + 1.0) * eps2
    ncv1 = int(np.floor((v1_max - v1_origin + (radius + 1.0) * eps1) / eps1)) + 2
    ncv2 = int(np.floor((v2_max - v2_origin + (radius + 1.0) * eps2) / eps2)) + 2
    if ncv1 < 1:
        ncv1 = 1
    if ncv2 < 1:
        ncv2 = 1

    head = np.empty(Nx * ncv1 * ncv2, dtype=np.int64)
    for i in range(head.shape[0]):
        head[i] = -1
    nxt = np.empty(N, dtype=np.int64)

    for p in range(N):
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
        cell = (ix * ncv1 + iv1) * ncv2 + iv2
        nxt[p] = head[cell]
        head[cell] = p

    return head, nxt, v1_origin, v2_origin, ncv1, ncv2


@njit(cache=True)
def _build_x_batch_cell_list(x, batch_ids, Lx, eta, Nx, R_eff):
    N = x.shape[0]
    head = np.empty(Nx * R_eff, dtype=np.int64)
    for i in range(head.shape[0]):
        head[i] = -1
    nxt = np.empty(N, dtype=np.int64)
    for p in range(N):
        ix = int(np.floor(x[p] / eta))
        if ix < 0 or ix >= Nx:
            ix = ix % Nx
        b = batch_ids[p]
        if b < 0:
            b = 0
        elif b >= R_eff:
            b = R_eff - 1
        cell = ix * R_eff + b
        nxt[p] = head[cell]
        head[cell] = p
    return head, nxt


@njit(parallel=True, fastmath=True, cache=True)
def _compute_ft_grad_pass(x, v1, v2, w, head, nxt, Lx, eta, eps1, eps2,
                          Nx, order, radius, v1_origin, v2_origin, ncv1, ncv2):
    N = x.shape[0]
    ft = np.empty(N, dtype=np.float64)
    g1 = np.empty(N, dtype=np.float64)
    g2 = np.empty(N, dtype=np.float64)
    rx = int(np.ceil(radius))
    rv1 = int(np.ceil(radius))
    rv2 = int(np.ceil(radius))

    for p in prange(N):
        ix0 = int(np.floor(x[p] / eta))
        if ix0 < 0 or ix0 >= Nx:
            ix0 = ix0 % Nx
        iv10 = int(np.floor((v1[p] - v1_origin) / eps1))
        iv20 = int(np.floor((v2[p] - v2_origin) / eps2))
        if iv10 < 0:
            iv10 = 0
        elif iv10 >= ncv1:
            iv10 = ncv1 - 1
        if iv20 < 0:
            iv20 = 0
        elif iv20 >= ncv2:
            iv20 = ncv2 - 1

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
                    q = head[cell]
                    while q != -1:
                        dx = periodic_delta(x[p], x[q], Lx)
                        bx, _ = spline_value_derivative(dx / eta, order)
                        if bx != 0.0:
                            dv1 = v1[p] - v1[q]
                            dv2 = v2[p] - v2[q]
                            b1, db1 = spline_value_derivative(dv1 / eps1, order)
                            if b1 != 0.0 or db1 != 0.0:
                                b2, db2 = spline_value_derivative(dv2 / eps2, order)
                                if b2 != 0.0 or db2 != 0.0:
                                    wx = bx / eta
                                    phi = b1 * b2 / (eps1 * eps2)
                                    grad1 = db1 * b2 / (eps1 * eps1 * eps2)
                                    grad2 = b1 * db2 / (eps1 * eps2 * eps2)
                                    wq = w[q]
                                    s0 += wq * wx * phi
                                    s1 += wq * wx * grad1
                                    s2 += wq * wx * grad2
                        q = nxt[q]
        ft[p] = s0
        g1[p] = s1
        g2[p] = s2
    return ft, g1, g2


@njit(parallel=True, fastmath=True, cache=True)
def _compute_F_second_pass(x, v1, v2, w, ft, grad1, grad2, head, nxt,
                           Lx, eta, eps1, eps2, Nx, order, radius,
                           v1_origin, v2_origin, ncv1, ncv2):
    N = x.shape[0]
    F1 = np.empty(N, dtype=np.float64)
    F2 = np.empty(N, dtype=np.float64)
    rx = int(np.ceil(radius))
    rv1 = int(np.ceil(radius))
    rv2 = int(np.ceil(radius))

    for p in prange(N):
        ix0 = int(np.floor(x[p] / eta))
        if ix0 < 0 or ix0 >= Nx:
            ix0 = ix0 % Nx
        iv10 = int(np.floor((v1[p] - v1_origin) / eps1))
        iv20 = int(np.floor((v2[p] - v2_origin) / eps2))
        if iv10 < 0:
            iv10 = 0
        elif iv10 >= ncv1:
            iv10 = ncv1 - 1
        if iv20 < 0:
            iv20 = 0
        elif iv20 >= ncv2:
            iv20 = ncv2 - 1

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
                    q = head[cell]
                    while q != -1:
                        dx = periodic_delta(x[p], x[q], Lx)
                        bx, _ = spline_value_derivative(dx / eta, order)
                        if bx != 0.0:
                            dv1 = v1[p] - v1[q]
                            dv2 = v2[p] - v2[q]
                            b1, db1 = spline_value_derivative(dv1 / eps1, order)
                            if b1 != 0.0 or db1 != 0.0:
                                b2, db2 = spline_value_derivative(dv2 / eps2, order)
                                if b2 != 0.0 or db2 != 0.0:
                                    wx = bx / eta
                                    gradv1 = db1 * b2 / (eps1 * eps1 * eps2)
                                    gradv2 = b1 * db2 / (eps1 * eps2 * eps2)
                                    coef = w[q] / ft[q]
                                    s1 += coef * wx * gradv1
                                    s2 += coef * wx * gradv2
                        q = nxt[q]
        F1[p] = grad1[p] / ft[p] + s1
        F2[p] = grad2[p] / ft[p] + s2
    return F1, F2


@njit(parallel=True, fastmath=True, cache=True)
def _compute_pair_accel_xbatch(xr, vr1, vr2, Fr1m, Fr2m, batch_r,
                               xs, vs1, vs2, ws, Fs1m, Fs2m,
                               head_src, nxt_src,
                               Lx, eta, Nx, order, radius, R_eff,
                               batch_scale, gam, Bcoef, same_species):
    Nr = xr.shape[0]
    a1 = np.empty(Nr, dtype=np.float64)
    a2 = np.empty(Nr, dtype=np.float64)
    rx = int(np.ceil(radius))
    half_gam = 0.5 * gam

    for p in prange(Nr):
        ix0 = int(np.floor(xr[p] / eta))
        if ix0 < 0 or ix0 >= Nx:
            ix0 = ix0 % Nx
        b = batch_r[p]
        if b < 0:
            b = 0
        elif b >= R_eff:
            b = R_eff - 1

        xp = xr[p]
        v1p = vr1[p]
        v2p = vr2[p]
        F1p = Fr1m[p]
        F2p = Fr2m[p]
        s1 = 0.0
        s2 = 0.0

        for dix in range(-rx, rx + 1):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            cell = ix * R_eff + b
            q = head_src[cell]
            while q != -1:
                if (not same_species) or (q != p):
                    dx = periodic_delta(xp, xs[q], Lx)
                    bx, _ = spline_value_derivative(dx / eta, order)
                    if bx != 0.0:
                        dv1 = v1p - vs1[q]
                        dv2 = v2p - vs2[q]
                        r2 = dv1 * dv1 + dv2 * dv2
                        if r2 > 0.0:
                            coeff = Bcoef * (r2 ** half_gam)
                            A11 = coeff * dv2 * dv2
                            A12 = -coeff * dv1 * dv2
                            A22 = coeff * dv1 * dv1
                            # acceleration = -U = sum A*(F_source/m_source - F_recv/m_recv)
                            bF1 = Fs1m[q] - F1p
                            bF2 = Fs2m[q] - F2p
                            psix = bx / eta
                            wq = ws[q] * psix * batch_scale
                            s1 += wq * (A11 * bF1 + A12 * bF2)
                            s2 += wq * (A12 * bF1 + A22 * bF2)
                q = nxt_src[q]
        a1[p] = s1
        a2[p] = s2
    return a1, a2


def _compute_species_F(x, v1, v2, w, eps1, eps2, par):
    radius = par.spline_radius
    head, nxt, v1_origin, v2_origin, ncv1, ncv2 = _build_phase_cell_list(
        x, v1, v2, par.Lx, par.eta, eps1, eps2, par.Nx, radius
    )
    ft, grad1, grad2 = _compute_ft_grad_pass(
        x, v1, v2, w, head, nxt, par.Lx, par.eta, eps1, eps2,
        par.Nx, par.spline_order, radius, v1_origin, v2_origin, ncv1, ncv2
    )
    F1, F2 = _compute_F_second_pass(
        x, v1, v2, w, ft, grad1, grad2, head, nxt,
        par.Lx, par.eta, eps1, eps2, par.Nx, par.spline_order, radius,
        v1_origin, v2_origin, ncv1, ncv2
    )
    return ft, F1, F2


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

    head1_xb, nxt1_xb = _build_x_batch_cell_list(x1, batch1, par.Lx, par.eta, par.Nx, R_eff)
    head2_xb, nxt2_xb = _build_x_batch_cell_list(x2, batch2, par.Lx, par.eta, par.Nx, R_eff)

    C = par.collision_strength
    # Species 1: self B11 and source species 2 B21.
    a11_self, a12_self = _compute_pair_accel_xbatch(
        x1, v11, v12, F11m, F12m, batch1,
        x1, v11, v12, w1, F11m, F12m,
        head1_xb, nxt1_xb, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius,
        R_eff, scale_same_1, par.gam, C * par.B11, True
    )
    a11_cross, a12_cross = _compute_pair_accel_xbatch(
        x1, v11, v12, F11m, F12m, batch1,
        x2, v21, v22, w2, F21m, F22m,
        head2_xb, nxt2_xb, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius,
        R_eff, scale_cross, par.gam, C * par.B21, False
    )

    # Species 2: self B22 and source species 1 B12.
    a21_self, a22_self = _compute_pair_accel_xbatch(
        x2, v21, v22, F21m, F22m, batch2,
        x2, v21, v22, w2, F21m, F22m,
        head2_xb, nxt2_xb, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius,
        R_eff, scale_same_2, par.gam, C * par.B22, True
    )
    a21_cross, a22_cross = _compute_pair_accel_xbatch(
        x2, v21, v22, F21m, F22m, batch2,
        x1, v11, v12, w1, F11m, F12m,
        head1_xb, nxt1_xb, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius,
        R_eff, scale_cross, par.gam, C * par.B12, False
    )

    return (
        a11_self + a11_cross,
        a12_self + a12_cross,
        a21_self + a21_cross,
        a22_self + a22_cross,
        ft1, ft2, F11, F12, F21, F22,
    )


def compute_regularized_density_at_particles(x1, v11, v12, w1, x2, v21, v22, w2, par):
    """Compute ftilde_i at particles for diagnostics."""
    ft1, _, _ = _compute_species_F(x1, v11, v12, w1, par.eps1_v1, par.eps1_v2, par)
    ft2, _, _ = _compute_species_F(x2, v21, v22, w2, par.eps2_v1, par.eps2_v2, par)
    return ft1, ft2
