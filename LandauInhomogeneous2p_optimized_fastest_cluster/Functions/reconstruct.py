"""
Reconstruction of the velocity marginals

    f_s(x, v1) = int f_s(x, v1, v2) dv2,      f_s(x, v2) = int f_s(x, v1, v2) dv1

for each species s, on a regular (x, v) grid.  f_s(x, v1, v2) itself is 3D
and is deliberately never assembled: each particle is deposited onto a 2D
(x, v_c) grid using the *same* compact B-spline mollifier as the
charge/current deposition in fields.py (periodic in x, clamped/non-periodic
in v_c), and the other velocity component is simply never used in that
deposition -- summing weight over all particles landing in a given (x, v_c)
bin, regardless of their v_c' value, already integrates f over v_c'.

This is diagnostic-only code (called at most every par.dist_save_interval,
not every step), but it reuses the same chunked-accumulation parallelisation
pattern as fields.py's charge/current deposition for consistency and so it
scales to the same particle counts.
"""
import numpy as np
from numba import njit, prange

from .splines_numba import spline_value_derivative, periodic_delta
from .cell_sort import default_nchunks


def velocity_grid(Lv, Nv):
    """Cell centres of a symmetric, non-periodic velocity grid on [-Lv, Lv],
    with bin width hv = 2*Lv/Nv acting as both the grid spacing and the
    mollifier bandwidth in the v direction (mirroring how eta plays both
    roles for the x-grid in fields.py)."""
    hv = 2.0 * Lv / Nv
    v = -Lv + (np.arange(Nv, dtype=float) + 0.5) * hv
    return v, hv


@njit(parallel=True, fastmath=True, cache=True)
def _deposit_marginal_chunked(x, vc, w, Lx, eta, Nx, order, radius,
                               Lv, Nv, hv, nchunks):
    N = x.shape[0]
    nchunks = max(1, min(nchunks, max(1, N)))
    chunk_size = (N + nchunks - 1) // nchunks
    r = int(np.ceil(radius)) + 2
    inv_eta = 1.0 / eta
    inv_hv = 1.0 / hv

    f_local = np.zeros((nchunks, Nx, Nv), dtype=np.float64)

    for c in prange(nchunks):
        start = c * chunk_size
        end = start + chunk_size
        if end > N:
            end = N
        for p in range(start, end):
            basex = int(np.floor(x[p] * inv_eta - 0.5))
            basev = int(np.floor((vc[p] + Lv) * inv_hv - 0.5))
            for offx in range(-r, r + 1):
                ix = (basex + offx) % Nx
                xg = (ix + 0.5) * eta
                dx = periodic_delta(xg, x[p], Lx)
                bx, _ = spline_value_derivative(dx * inv_eta, order)
                if bx == 0.0:
                    continue
                for offv in range(-r, r + 1):
                    iv = basev + offv
                    # Non-periodic: contributions that fall outside the
                    # velocity domain are dropped (particles are already
                    # truncated to it when par.truncate_velocity_samples).
                    if iv < 0 or iv >= Nv:
                        continue
                    vg = -Lv + (iv + 0.5) * hv
                    dv = vc[p] - vg
                    bv, _ = spline_value_derivative(dv * inv_hv, order)
                    if bv == 0.0:
                        continue
                    f_local[c, ix, iv] += w[p] * bx * inv_eta * bv * inv_hv

    f = np.zeros((Nx, Nv), dtype=np.float64)
    for c in range(nchunks):
        f += f_local[c]
    return f


def deposit_marginal(x, vc, w, Lv, Nv, par):
    """f_tilde(x, v_c) on an (Nx, Nv) grid, for one species/component."""
    v, hv = velocity_grid(Lv, Nv)
    nchunks = default_nchunks(x.shape[0], par.sort_oversample)
    f = _deposit_marginal_chunked(
        x, vc, w, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius,
        Lv, Nv, hv, nchunks,
    )
    return f, v


def compute_all_marginals(x1, v11, v12, w1, x2, v21, v22, w2, par):
    """Reconstruct f1(x,v1), f1(x,v2), f2(x,v1), f2(x,v2) at the current
    particle state.  Grid *domains* use par.dist_Lv1_v1/dist_Lv1_v2/
    dist_Lv2_v1/dist_Lv2_v2 (default: the physics velocity-sampling
    domain par.Lv1_v1/etc, but independently overridable -- see
    params.py); grid *resolutions* use par.dist_Nv1_v1/dist_Nv1_v2/
    dist_Nv2_v1/dist_Nv2_v2, which default to the physics Nv1_v1/etc, the
    same numbers that set the particle count. Both are purely diagnostic:
    neither affects particle sampling, the field solver, or the collision
    operator.

    Returns a dict with the four (Nx, Nv) arrays and the grids used.
    """
    f1_vx, v1_v1_grid = deposit_marginal(x1, v11, w1, par.dist_Lv1_v1, par.dist_Nv1_v1, par)
    f1_vy, v1_v2_grid = deposit_marginal(x1, v12, w1, par.dist_Lv1_v2, par.dist_Nv1_v2, par)
    f2_vx, v2_v1_grid = deposit_marginal(x2, v21, w2, par.dist_Lv2_v1, par.dist_Nv2_v1, par)
    f2_vy, v2_v2_grid = deposit_marginal(x2, v22, w2, par.dist_Lv2_v2, par.dist_Nv2_v2, par)
    return {
        "f1_vx": f1_vx, "f1_vy": f1_vy,
        "f2_vx": f2_vx, "f2_vy": f2_vy,
        "x_grid": par.x_grid.copy(),
        "v1_v1_grid": v1_v1_grid, "v1_v2_grid": v1_v2_grid,
        "v2_v1_grid": v2_v1_grid, "v2_v2_grid": v2_v2_grid,
    }


@njit(parallel=True, fastmath=True, cache=True)
def _deposit_velocity_marginal_chunked(v1, v2, w, order, radius,
                                        Lv1, Nv1, hv1, Lv2, Nv2, hv2, nchunks):
    N = v1.shape[0]
    nchunks = max(1, min(nchunks, max(1, N)))
    chunk_size = (N + nchunks - 1) // nchunks
    r = int(np.ceil(radius)) + 2
    inv_hv1 = 1.0 / hv1
    inv_hv2 = 1.0 / hv2

    f_local = np.zeros((nchunks, Nv1, Nv2), dtype=np.float64)

    for c in prange(nchunks):
        start = c * chunk_size
        end = start + chunk_size
        if end > N:
            end = N
        for p in range(start, end):
            base1 = int(np.floor((v1[p] + Lv1) * inv_hv1 - 0.5))
            base2 = int(np.floor((v2[p] + Lv2) * inv_hv2 - 0.5))
            for off1 in range(-r, r + 1):
                i1 = base1 + off1
                # Non-periodic in both directions: out-of-domain
                # contributions are dropped, exactly as in
                # _deposit_marginal_chunked's v-direction handling above.
                if i1 < 0 or i1 >= Nv1:
                    continue
                vg1 = -Lv1 + (i1 + 0.5) * hv1
                d1 = v1[p] - vg1
                b1, _ = spline_value_derivative(d1 * inv_hv1, order)
                if b1 == 0.0:
                    continue
                for off2 in range(-r, r + 1):
                    i2 = base2 + off2
                    if i2 < 0 or i2 >= Nv2:
                        continue
                    vg2 = -Lv2 + (i2 + 0.5) * hv2
                    d2 = v2[p] - vg2
                    b2, _ = spline_value_derivative(d2 * inv_hv2, order)
                    if b2 == 0.0:
                        continue
                    f_local[c, i1, i2] += w[p] * b1 * inv_hv1 * b2 * inv_hv2

    f = np.zeros((Nv1, Nv2), dtype=np.float64)
    for c in range(nchunks):
        f += f_local[c]
    return f


def deposit_velocity_marginal(v1, v2, w, Lv1, Nv1, Lv2, Nv2, par):
    """f_tilde(v1, v2) = int f_tilde(x, v1, v2) dx, on an (Nv1, Nv2) grid,
    for one species -- the full 2D velocity-space marginal used e.g. to
    visualise beam collapse in the Weibel test (Bailo, Carrillo & Hu 2024,
    Figs 9-10). Same compact-B-spline reconstruction as deposit_marginal
    above, just dropping the (periodic) x dimension entirely: every
    particle contributes to the (v1,v2) grid regardless of its position."""
    v1_grid, hv1 = velocity_grid(Lv1, Nv1)
    v2_grid, hv2 = velocity_grid(Lv2, Nv2)
    nchunks = default_nchunks(v1.shape[0], par.sort_oversample)
    f = _deposit_velocity_marginal_chunked(
        v1, v2, w, par.spline_order, par.spline_radius,
        Lv1, Nv1, hv1, Lv2, Nv2, hv2, nchunks,
    )
    return f, v1_grid, v2_grid


def compute_velocity_marginals(v11, v12, w1, v21, v22, w2, par):
    """Reconstruct f1(v1,v2), f2(v1,v2) (integrated over x) at the current
    particle state. Grid *domains* use par.dist_Lv1_v1/dist_Lv1_v2/
    dist_Lv2_v1/dist_Lv2_v2 (default: the physics domain, independently
    overridable); grid *resolutions* use par.dist_Nv1_v1/etc -- exactly
    as compute_all_marginals does for the (x,v) marginals above.
    """
    f1_v1v2, v1_v1_grid, v1_v2_grid = deposit_velocity_marginal(
        v11, v12, w1, par.dist_Lv1_v1, par.dist_Nv1_v1, par.dist_Lv1_v2, par.dist_Nv1_v2, par)
    f2_v1v2, v2_v1_grid, v2_v2_grid = deposit_velocity_marginal(
        v21, v22, w2, par.dist_Lv2_v1, par.dist_Nv2_v1, par.dist_Lv2_v2, par.dist_Nv2_v2, par)
    return {
        "f1_v1v2": f1_v1v2, "f2_v1v2": f2_v1v2,
        "v1_v1_grid": v1_v1_grid, "v1_v2_grid": v1_v2_grid,
        "v2_v1_grid": v2_v1_grid, "v2_v2_grid": v2_v2_grid,
    }


class VelocityMarginalLog:
    """
    Snapshots of the reconstructed velocity-only marginals f_s(v1,v2)
    (integrated over x), taken every par.dist_save_interval by a main
    driver's time_cycle. Separate from DistributionLog (which reconstructs
    f_s(x,v_c)) since not every test needs both -- currently only
    main_Weibel_2species.py uses this one, to visualise beam collapse as
    in Bailo, Carrillo & Hu (2024), Figs 9-10.
    """
    def __init__(self):
        self.t = []
        self.f1_v1v2 = []
        self.f2_v1v2 = []
        self._grids = None

    def append(self, t, v11, v12, w1, v21, v22, w2, par):
        m = compute_velocity_marginals(v11, v12, w1, v21, v22, w2, par)
        if self._grids is None:
            self._grids = {
                "v1_v1_grid": m["v1_v1_grid"], "v1_v2_grid": m["v1_v2_grid"],
                "v2_v1_grid": m["v2_v1_grid"], "v2_v2_grid": m["v2_v2_grid"],
            }
        self.t.append(float(t))
        self.f1_v1v2.append(m["f1_v1v2"])
        self.f2_v1v2.append(m["f2_v1v2"])

    def as_dict(self):
        empty_grid = np.zeros((0,))
        out = {
            "t": np.asarray(self.t),
            "f1_v1v2": np.stack(self.f1_v1v2, axis=0) if self.f1_v1v2 else np.zeros((0, 0, 0)),
            "f2_v1v2": np.stack(self.f2_v1v2, axis=0) if self.f2_v1v2 else np.zeros((0, 0, 0)),
        }
        grids = self._grids or {}
        for name in ("v1_v1_grid", "v1_v2_grid", "v2_v1_grid", "v2_v2_grid"):
            out[name] = grids.get(name, empty_grid)
        return out


class DistributionLog:
    """
    Snapshots of the reconstructed velocity marginals f_s(x, v_c), taken
    every par.dist_save_interval by the main driver's time_cycle.

    Grids (x_grid, v*_grid) are fixed for the whole run and are stored
    once; only the four marginal arrays are stacked per snapshot, giving
    arrays of shape (n_snapshots, Nx, Nv) in as_dict().
    """
    def __init__(self):
        self.t = []
        self.f1_vx = []
        self.f1_vy = []
        self.f2_vx = []
        self.f2_vy = []
        self._grids = None

    def append(self, t, x1, v11, v12, w1, x2, v21, v22, w2, par):
        m = compute_all_marginals(x1, v11, v12, w1, x2, v21, v22, w2, par)
        if self._grids is None:
            self._grids = {
                "x_grid": m["x_grid"],
                "v1_v1_grid": m["v1_v1_grid"], "v1_v2_grid": m["v1_v2_grid"],
                "v2_v1_grid": m["v2_v1_grid"], "v2_v2_grid": m["v2_v2_grid"],
            }
        self.t.append(float(t))
        self.f1_vx.append(m["f1_vx"])
        self.f1_vy.append(m["f1_vy"])
        self.f2_vx.append(m["f2_vx"])
        self.f2_vy.append(m["f2_vy"])

    def as_dict(self):
        empty_grid = np.zeros((0,))
        out = {
            "t": np.asarray(self.t),
            "f1_vx": np.stack(self.f1_vx, axis=0) if self.f1_vx else np.zeros((0, 0, 0)),
            "f1_vy": np.stack(self.f1_vy, axis=0) if self.f1_vy else np.zeros((0, 0, 0)),
            "f2_vx": np.stack(self.f2_vx, axis=0) if self.f2_vx else np.zeros((0, 0, 0)),
            "f2_vy": np.stack(self.f2_vy, axis=0) if self.f2_vy else np.zeros((0, 0, 0)),
        }
        grids = self._grids or {}
        for name in ("x_grid", "v1_v1_grid", "v1_v2_grid", "v2_v1_grid", "v2_v2_grid"):
            out[name] = grids.get(name, empty_grid)
        return out
