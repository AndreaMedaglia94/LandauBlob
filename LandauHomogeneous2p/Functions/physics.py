"""
Blocked NumPy implementation of the two-species homogeneous Landau RHS.

The update follows the MATLAB BKW_2D_pulito code:

    dv_i/dt = -(1/m_i) sum_j sum_q w_jq A_ji(v_ip - v_jq)
                         [G_i(v_ip)/m_i - G_j(v_jq)/m_j],

where G_i is the regularized entropy-gradient vector.  compute_F_over_m()
returns G_i/m_i, exactly like the MATLAB FF.m routine.
"""
import numpy as np
from .kernels import PSI_KERNELS, GRAD_KERNELS, pairwise_diffs


def _blocks(n, block_size):
    for start in range(0, n, block_size):
        yield slice(start, min(start + block_size, n))


def compute_F_over_m(vx, vy, wg, mass, epsi, shape, block_size=1000):
    """Return G(v_p)/mass for every particle of one species."""
    psi_kernel = PSI_KERNELS[shape]
    grad_kernel = GRAD_KERNELS[shape]
    n = vx.shape[0]
    ft = np.empty(n)
    gxw = np.empty(n)
    gyw = np.empty(n)

    for blk in _blocks(n, block_size):
        dx, dy = pairwise_diffs(vx[blk], vy[blk], vx, vy)
        psi = psi_kernel(dx, dy, epsi)
        gx, gy = grad_kernel(dx, dy, epsi)
        ft[blk] = psi @ wg
        gxw[blk] = gx @ wg
        gyw[blk] = gy @ wg

    w_over_ft = wg / ft
    Fx = np.empty(n)
    Fy = np.empty(n)
    for blk in _blocks(n, block_size):
        dx, dy = pairwise_diffs(vx[blk], vy[blk], vx, vy)
        gx, gy = grad_kernel(dx, dy, epsi)
        Fx[blk] = gxw[blk] / ft[blk] + gx @ w_over_ft
        Fy[blk] = gyw[blk] / ft[blk] + gy @ w_over_ft

    return Fx / mass, Fy / mass


def _pair_contribution(vx_i, vy_i, vx_j, vy_j, wg_j, Fx_i, Fy_i, Fx_j, Fy_j,
                       mass_i, beta_ji, gam, block_size):
    """Contribution to species i from source species j."""
    n_i = vx_i.shape[0]
    dvx = np.zeros(n_i)
    dvy = np.zeros(n_i)
    half_gam = 0.5 * gam

    for blk in _blocks(n_i, block_size):
        dx, dy = pairwise_diffs(vx_i[blk], vy_i[blk], vx_j, vy_j)
        r2 = dx**2 + dy**2
        with np.errstate(divide="ignore", invalid="ignore"):
            nn = np.where(r2 > 0.0, beta_ji * r2**half_gam, 0.0)
        A11 = nn * dy**2
        A12 = -nn * dx * dy
        A22 = nn * dx**2

        dFx = Fx_i[blk, None] - Fx_j[None, :]
        dFy = Fy_i[blk, None] - Fy_j[None, :]

        sx = (A11 * dFx + A12 * dFy) @ wg_j
        sy = (A12 * dFx + A22 * dFy) @ wg_j
        dvx[blk] = -sx / mass_i
        dvy[blk] = -sy / mass_i

    return dvx, dvy


def compute_rhs(vx1, vy1, wg1, vx2, vy2, wg2, par):
    """Return dv/dt for both species: (dvx1, dvy1, dvx2, dvy2)."""
    Fx1, Fy1 = compute_F_over_m(vx1, vy1, wg1, par.m1, par.epsi1, par.shape, par.block_size)
    Fx2, Fy2 = compute_F_over_m(vx2, vy2, wg2, par.m2, par.epsi2, par.shape, par.block_size)

    dvx11, dvy11 = _pair_contribution(vx1, vy1, vx1, vy1, wg1, Fx1, Fy1, Fx1, Fy1,
                                      par.m1, par.B11, par.gam, par.block_size)
    dvx21, dvy21 = _pair_contribution(vx1, vy1, vx2, vy2, wg2, Fx1, Fy1, Fx2, Fy2,
                                      par.m1, par.B21, par.gam, par.block_size)
    dvx22, dvy22 = _pair_contribution(vx2, vy2, vx2, vy2, wg2, Fx2, Fy2, Fx2, Fy2,
                                      par.m2, par.B22, par.gam, par.block_size)
    dvx12, dvy12 = _pair_contribution(vx2, vy2, vx1, vy1, wg1, Fx2, Fy2, Fx1, Fy1,
                                      par.m2, par.B12, par.gam, par.block_size)

    return dvx11 + dvx21, dvy11 + dvy21, dvx22 + dvx12, dvy22 + dvy12
