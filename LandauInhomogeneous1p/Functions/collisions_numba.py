"""
Numba implementation of the one-species inhomogeneous C-PIC collision force
for the 1D-2V setting.

The implementation follows the three steps in section 2.2.3 of the C-PIC paper:
  I   compute f_tilde and grad_v f_tilde at particles using a phase-space cell list;
  II  compute F = grad_v(delta H/delta f) at particles using the same cell list;
  III compute the Landau effective acceleration -U, localised in x and optionally
       accelerated by random batching.

Only compact B-splines of order 1, 2 and 3 are supported.
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
def _build_phase_cell_list(x, v1, v2, Lx, eta, eps1, eps2, Nx, order, radius):
    N = x.shape[0]
    v1_min, v1_max = _minmax(v1)
    v2_min, v2_max = _minmax(v2)

    # Add a support-sized margin so all neighbour searches remain in range.
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
        if ix < 0:
            ix = ix % Nx
        elif ix >= Nx:
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
        if ix < 0:
            ix = ix % Nx
        elif ix >= Nx:
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
        if ix0 < 0:
            ix0 = ix0 % Nx
        elif ix0 >= Nx:
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
        if ix0 < 0:
            ix0 = ix0 % Nx
        elif ix0 >= Nx:
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
def _compute_collision_accel_pass(x, v1, v2, w, F1, F2, batch_ids,
                                  head_xb, nxt_xb, Lx, eta, Nx, order, radius,
                                  R_eff, batch_scale, gam, C):
    N = x.shape[0]
    a1 = np.empty(N, dtype=np.float64)
    a2 = np.empty(N, dtype=np.float64)
    rx = int(np.ceil(radius))
    half_gam = 0.5 * gam

    for p in prange(N):
        ix0 = int(np.floor(x[p] / eta))
        if ix0 < 0:
            ix0 = ix0 % Nx
        elif ix0 >= Nx:
            ix0 = ix0 % Nx
        b = batch_ids[p]
        if b < 0:
            b = 0
        elif b >= R_eff:
            b = R_eff - 1

        s1 = 0.0
        s2 = 0.0
        xp = x[p]
        v1p = v1[p]
        v2p = v2[p]
        F1p = F1[p]
        F2p = F2[p]

        for dix in range(-rx, rx + 1):
            ix = ix0 + dix
            if ix < 0:
                ix += Nx
            elif ix >= Nx:
                ix -= Nx
            cell = ix * R_eff + b
            q = head_xb[cell]
            while q != -1:
                if q != p:
                    dx = periodic_delta(xp, x[q], Lx)
                    bx, _ = spline_value_derivative(dx / eta, order)
                    if bx != 0.0:
                        dv1 = v1p - v1[q]
                        dv2 = v2p - v2[q]
                        r2 = dv1 * dv1 + dv2 * dv2
                        if r2 > 0.0:
                            coeff = C * (r2 ** half_gam)
                            A11 = coeff * dv2 * dv2
                            A12 = -coeff * dv1 * dv2
                            A22 = coeff * dv1 * dv1
                            bF1 = F1[q] - F1p
                            bF2 = F2[q] - F2p
                            psix = bx / eta
                            wq = w[q] * psix * batch_scale
                            s1 += wq * (A11 * bF1 + A12 * bF2)
                            s2 += wq * (A12 * bF1 + A22 * bF2)
                q = nxt_xb[q]
        a1[p] = s1
        a2[p] = s2

    return a1, a2


def compute_collision_acceleration(x, v1, v2, w, batch_ids, par):
    """
    Return (a1,a2,ft,F1,F2), where (a1,a2) is the collisional acceleration
    -U_eta,eps[f^N] acting in velocity.

    If par.enable_collisions is False or par.collision_strength is zero, the
    returned acceleration is zero and ft/F are still computed only when needed
    by diagnostics outside this function.
    """
    if (not par.enable_collisions) or par.collision_strength == 0.0:
        z = np.zeros_like(v1)
        return z, z.copy(), None, None, None

    radius = par.spline_radius
    head, nxt, v1_origin, v2_origin, ncv1, ncv2 = _build_phase_cell_list(
        x, v1, v2, par.Lx, par.eta, par.eps1, par.eps2, par.Nx, par.spline_order, radius
    )
    ft, grad1, grad2 = _compute_ft_grad_pass(
        x, v1, v2, w, head, nxt, par.Lx, par.eta, par.eps1, par.eps2,
        par.Nx, par.spline_order, radius, v1_origin, v2_origin, ncv1, ncv2
    )
    F1, F2 = _compute_F_second_pass(
        x, v1, v2, w, ft, grad1, grad2, head, nxt, par.Lx, par.eta, par.eps1, par.eps2,
        par.Nx, par.spline_order, radius, v1_origin, v2_origin, ncv1, ncv2
    )

    if par.random_batch and par.random_batches > 1:
        R_eff = par.random_batches
        batch_scale = R_eff * (x.shape[0] - 1.0) / (x.shape[0] - R_eff)
    else:
        R_eff = 1
        batch_scale = 1.0

    head_xb, nxt_xb = _build_x_batch_cell_list(x, batch_ids, par.Lx, par.eta, par.Nx, R_eff)
    a1, a2 = _compute_collision_accel_pass(
        x, v1, v2, w, F1, F2, batch_ids, head_xb, nxt_xb,
        par.Lx, par.eta, par.Nx, par.spline_order, radius, R_eff,
        batch_scale, par.gam, par.collision_strength
    )
    return a1, a2, ft, F1, F2


def compute_regularized_density_at_particles(x, v1, v2, w, par):
    """Compute f_tilde at particles for diagnostics without the collision force."""
    radius = par.spline_radius
    head, nxt, v1_origin, v2_origin, ncv1, ncv2 = _build_phase_cell_list(
        x, v1, v2, par.Lx, par.eta, par.eps1, par.eps2, par.Nx, par.spline_order, radius
    )
    ft, _, _ = _compute_ft_grad_pass(
        x, v1, v2, w, head, nxt, par.Lx, par.eta, par.eps1, par.eps2,
        par.Nx, par.spline_order, radius, v1_origin, v2_origin, ncv1, ncv2
    )
    return ft
