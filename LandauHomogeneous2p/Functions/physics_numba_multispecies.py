"""
Parallel Numba RHS for the two-species homogeneous Landau particle code.

This is the default fast implementation used by main_BKW_multispecies.py.
It computes the same RHS as Functions.physics.compute_rhs but avoids large
N-by-N temporaries.  For gam == 0 it uses Maxwell-molecule moment identities;
for gam != 0 it evaluates the direct pair sums in parallel over receiver
particles.
"""
import numpy as np
from numba import njit, prange


@njit(inline="always", fastmath=True)
def _abs(x):
    return x if x >= 0.0 else -x


@njit(inline="always", fastmath=True)
def _sign(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@njit(inline="always", fastmath=True)
def _psi_grad(dx, dy, epsi, shape):
    if shape == 0:
        psi = np.exp(-(dx * dx + dy * dy) / (2.0 * epsi)) / (2.0 * np.pi * epsi)
        gx = -dx / epsi * psi
        gy = -dy / epsi * psi
        return psi, gx, gy

    X = dx / epsi
    Y = dy / epsi
    aX = _abs(X)
    aY = _abs(Y)

    if shape == 1:
        Gx = 1.0 - aX if aX <= 1.0 else 0.0
        Gy = 1.0 - aY if aY <= 1.0 else 0.0
        psi = Gx * Gy / (epsi * epsi)
        dGx = -_sign(X) if aX <= 1.0 else 0.0
        dGy = -_sign(Y) if aY <= 1.0 else 0.0
        gx = Gy * dGx / (epsi * epsi * epsi)
        gy = Gx * dGy / (epsi * epsi * epsi)
        return psi, gx, gy

    if shape == 2:
        if aX <= 0.5:
            Gx = 0.75 - X * X
            dGx = -2.0 * X
        elif aX <= 1.5:
            t = 1.5 - aX
            Gx = 0.5 * t * t
            dGx = -_sign(X) * t
        else:
            Gx = 0.0
            dGx = 0.0

        if aY <= 0.5:
            Gy = 0.75 - Y * Y
            dGy = -2.0 * Y
        elif aY <= 1.5:
            t = 1.5 - aY
            Gy = 0.5 * t * t
            dGy = -_sign(Y) * t
        else:
            Gy = 0.0
            dGy = 0.0

        psi = Gx * Gy / (epsi * epsi)
        gx = Gy * dGx / (epsi * epsi * epsi)
        gy = Gx * dGy / (epsi * epsi * epsi)
        return psi, gx, gy

    # shape == 3
    if aX <= 1.0:
        Gx = (4.0 - 6.0 * X * X + 3.0 * aX * aX * aX) / 6.0
        dGx = (-12.0 * X + 9.0 * X * aX) / 6.0
    elif aX <= 2.0:
        t = 2.0 - aX
        Gx = t * t * t / 6.0
        dGx = -0.5 * t * t * _sign(X)
    else:
        Gx = 0.0
        dGx = 0.0

    if aY <= 1.0:
        Gy = (4.0 - 6.0 * Y * Y + 3.0 * aY * aY * aY) / 6.0
        dGy = (-12.0 * Y + 9.0 * Y * aY) / 6.0
    elif aY <= 2.0:
        t = 2.0 - aY
        Gy = t * t * t / 6.0
        dGy = -0.5 * t * t * _sign(Y)
    else:
        Gy = 0.0
        dGy = 0.0

    psi = Gx * Gy / (epsi * epsi)
    gx = Gy * dGx / (epsi * epsi * epsi)
    gy = Gx * dGy / (epsi * epsi * epsi)
    return psi, gx, gy


@njit(parallel=True, fastmath=True, cache=True)
def _compute_ft_and_grad_w(vx, vy, wg, epsi, shape):
    N = vx.shape[0]
    ft = np.empty(N)
    gxw = np.empty(N)
    gyw = np.empty(N)

    for i in prange(N):
        xi = vx[i]
        yi = vy[i]
        s0 = 0.0
        sx = 0.0
        sy = 0.0
        for j in range(N):
            psi, gx, gy = _psi_grad(xi - vx[j], yi - vy[j], epsi, shape)
            w = wg[j]
            s0 += psi * w
            sx += gx * w
            sy += gy * w
        ft[i] = s0
        gxw[i] = sx
        gyw[i] = sy
    return ft, gxw, gyw


@njit(parallel=True, fastmath=True, cache=True)
def _compute_F_over_m_second_pass(vx, vy, wg, ft, gxw, gyw, mass, epsi, shape):
    N = vx.shape[0]
    Fx = np.empty(N)
    Fy = np.empty(N)

    for i in prange(N):
        xi = vx[i]
        yi = vy[i]
        sx = 0.0
        sy = 0.0
        for j in range(N):
            _, gx, gy = _psi_grad(xi - vx[j], yi - vy[j], epsi, shape)
            q = wg[j] / ft[j]
            sx += gx * q
            sy += gy * q
        Fx[i] = (gxw[i] / ft[i] + sx) / mass
        Fy[i] = (gyw[i] / ft[i] + sy) / mass
    return Fx, Fy


@njit(fastmath=True, cache=True)
def _moments(q, vx, vy):
    m0 = 0.0
    mx = 0.0
    my = 0.0
    mxx = 0.0
    mxy = 0.0
    myy = 0.0
    for i in range(q.shape[0]):
        qi = q[i]
        x = vx[i]
        y = vy[i]
        m0 += qi
        mx += qi * x
        my += qi * y
        mxx += qi * x * x
        mxy += qi * x * y
        myy += qi * y * y
    return m0, mx, my, mxx, mxy, myy


@njit(parallel=True, fastmath=True, cache=True)
def _dv_pair_maxwell(vx_i, vy_i, vx_j, vy_j, wg_j, Fx_i, Fy_i, Fx_j, Fy_j, beta_ji, mass_i):
    N_i = vx_i.shape[0]
    wFx = wg_j * Fx_j
    wFy = wg_j * Fy_j

    m0, mx, my, mxx, mxy, myy = _moments(wg_j, vx_j, vy_j)
    a0, ax, ay, axx, axy, ayy = _moments(wFx, vx_j, vy_j)
    b0, bx, by, bxx, bxy, byy = _moments(wFy, vx_j, vy_j)

    dvx = np.empty(N_i)
    dvy = np.empty(N_i)
    for i in prange(N_i):
        x = vx_i[i]
        y = vy_i[i]

        R11 = beta_ji * (m0 * y * y - 2.0 * y * my + myy)
        R12 = -beta_ji * (m0 * x * y - x * my - y * mx + mxy)
        R22 = beta_ji * (m0 * x * x - 2.0 * x * mx + mxx)

        T11 = beta_ji * (a0 * y * y - 2.0 * y * ay + ayy)
        T12x = -beta_ji * (a0 * x * y - x * ay - y * ax + axy)
        T12y = -beta_ji * (b0 * x * y - x * by - y * bx + bxy)
        T22 = beta_ji * (b0 * x * x - 2.0 * x * bx + bxx)

        dvx[i] = ((T11 - Fx_i[i] * R11) + (T12y - Fy_i[i] * R12)) / mass_i
        dvy[i] = ((T12x - Fx_i[i] * R12) + (T22 - Fy_i[i] * R22)) / mass_i
    return dvx, dvy


@njit(parallel=True, fastmath=True, cache=True)
def _dv_pair_direct(vx_i, vy_i, vx_j, vy_j, wg_j, Fx_i, Fy_i, Fx_j, Fy_j, beta_ji, gam, mass_i):
    N_i = vx_i.shape[0]
    N_j = vx_j.shape[0]
    dvx = np.empty(N_i)
    dvy = np.empty(N_i)
    half_gam = 0.5 * gam

    for i in prange(N_i):
        xi = vx_i[i]
        yi = vy_i[i]
        Fxi = Fx_i[i]
        Fyi = Fy_i[i]
        sx = 0.0
        sy = 0.0
        for j in range(N_j):
            dx = xi - vx_j[j]
            dy = yi - vy_j[j]
            r2 = dx * dx + dy * dy
            if r2 > 0.0:
                nn = beta_ji * (r2 ** half_gam)
                a11 = nn * dy * dy
                a12 = -nn * dx * dy
                a22 = nn * dx * dx
                dfx = Fxi - Fx_j[j]
                dfy = Fyi - Fy_j[j]
                w = wg_j[j]
                sx += w * (a11 * dfx + a12 * dfy)
                sy += w * (a12 * dfx + a22 * dfy)
        dvx[i] = -sx / mass_i
        dvy[i] = -sy / mass_i
    return dvx, dvy


def compute_F_over_m(vx, vy, wg, mass, epsi, shape):
    ft, gxw, gyw = _compute_ft_and_grad_w(vx, vy, wg, epsi, shape)
    return _compute_F_over_m_second_pass(vx, vy, wg, ft, gxw, gyw, mass, epsi, shape)


def _pair(vx_i, vy_i, vx_j, vy_j, wg_j, Fx_i, Fy_i, Fx_j, Fy_j, beta_ji, gam, mass_i):
    if gam == 0.0:
        return _dv_pair_maxwell(vx_i, vy_i, vx_j, vy_j, wg_j, Fx_i, Fy_i, Fx_j, Fy_j, beta_ji, mass_i)
    return _dv_pair_direct(vx_i, vy_i, vx_j, vy_j, wg_j, Fx_i, Fy_i, Fx_j, Fy_j, beta_ji, gam, mass_i)


def compute_rhs(vx1, vy1, wg1, vx2, vy2, wg2, par):
    Fx1, Fy1 = compute_F_over_m(vx1, vy1, wg1, par.m1, par.epsi1, par.shape)
    Fx2, Fy2 = compute_F_over_m(vx2, vy2, wg2, par.m2, par.epsi2, par.shape)

    dvx11, dvy11 = _pair(vx1, vy1, vx1, vy1, wg1, Fx1, Fy1, Fx1, Fy1, par.B11, par.gam, par.m1)
    dvx21, dvy21 = _pair(vx1, vy1, vx2, vy2, wg2, Fx1, Fy1, Fx2, Fy2, par.B21, par.gam, par.m1)
    dvx22, dvy22 = _pair(vx2, vy2, vx2, vy2, wg2, Fx2, Fy2, Fx2, Fy2, par.B22, par.gam, par.m2)
    dvx12, dvy12 = _pair(vx2, vy2, vx1, vy1, wg1, Fx2, Fy2, Fx1, Fy1, par.B12, par.gam, par.m2)

    return dvx11 + dvx21, dvy11 + dvy21, dvx22 + dvx12, dvy22 + dvy12
