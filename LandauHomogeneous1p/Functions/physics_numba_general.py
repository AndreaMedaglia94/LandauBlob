"""
General Numba RHS for the homogeneous 2D Landau particle code.

This file is meant to be imported by main.py as:
    from physics_numba_general import compute_rhs

Scope:
- Supports all current mollifier shapes: 0 Gaussian, 1/2/3 tensor B-splines.
- Supports arbitrary par.gam, including negative values; self-interaction r2=0 is skipped.
- Uses O(N) persistent memory and no NxN temporary arrays.
- For gam == 0 it uses the exact Maxwell-molecule moment identity for compute_dv.
- For gam != 0 it evaluates compute_dv by direct all-pairs summation, parallelized over i.

Important:
This is still the exact direct particle RHS. For general gam it is O(N^2) work per RHS call.
It is faster and more memory-stable than the blocked NumPy version for many cases, but it does
not make N=1e6 feasible without an algorithmic accelerator.
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
    """Return psi, dpsi/dx, dpsi/dy for one pairwise displacement."""
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
def _compute_F_second_pass(vx, vy, wg, ft, gxw, gyw, epsi, shape):
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
        Fx[i] = gxw[i] / ft[i] + sx
        Fy[i] = gyw[i] / ft[i] + sy
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
def _dv_maxwell(vx, vy, wg, Fx, Fy, beta):
    N = vx.shape[0]
    wFx = wg * Fx
    wFy = wg * Fy

    m0, mx, my, mxx, mxy, myy = _moments(wg, vx, vy)
    a0, ax, ay, axx, axy, ayy = _moments(wFx, vx, vy)
    b0, bx, by, bxx, bxy, byy = _moments(wFy, vx, vy)

    dvx = np.empty(N)
    dvy = np.empty(N)
    for i in prange(N):
        x = vx[i]
        y = vy[i]
        R11 = beta * (m0 * y * y - 2.0 * y * my + myy)
        R12 = -beta * (m0 * x * y - x * my - y * mx + mxy)
        R22 = beta * (m0 * x * x - 2.0 * x * mx + mxx)

        T11 = beta * (a0 * y * y - 2.0 * y * ay + ayy)
        T12x = -beta * (a0 * x * y - x * ay - y * ax + axy)
        T12y = -beta * (b0 * x * y - x * by - y * bx + bxy)
        T22 = beta * (b0 * x * x - 2.0 * x * bx + bxx)

        dvx[i] = (T11 - Fx[i] * R11) + (T12y - Fy[i] * R12)
        dvy[i] = (T12x - Fx[i] * R12) + (T22 - Fy[i] * R22)
    return dvx, dvy


@njit(parallel=True, fastmath=True, cache=True)
def _dv_direct(vx, vy, wg, Fx, Fy, beta, gam):
    N = vx.shape[0]
    dvx = np.empty(N)
    dvy = np.empty(N)
    half_gam = 0.5 * gam

    for i in prange(N):
        xi = vx[i]
        yi = vy[i]
        Fxi = Fx[i]
        Fyi = Fy[i]

        R11 = 0.0
        R12 = 0.0
        R22 = 0.0
        T11 = 0.0
        T12x = 0.0
        T12y = 0.0
        T22 = 0.0

        for j in range(N):
            dx = xi - vx[j]
            dy = yi - vy[j]
            r2 = dx * dx + dy * dy
            if r2 > 0.0:
                nn = beta * (r2 ** half_gam)
                a11 = nn * dy * dy
                a12 = -nn * dx * dy
                a22 = nn * dx * dx
                w = wg[j]
                wfx = w * Fx[j]
                wfy = w * Fy[j]
                R11 += a11 * w
                R12 += a12 * w
                R22 += a22 * w
                T11 += a11 * wfx
                T12x += a12 * wfx
                T12y += a12 * wfy
                T22 += a22 * wfy

        dvx[i] = (T11 - Fxi * R11) + (T12y - Fyi * R12)
        dvy[i] = (T12x - Fxi * R12) + (T22 - Fyi * R22)
    return dvx, dvy


def compute_f_tilde(vx, vy, wg, par):
    ft, _, _ = _compute_ft_and_grad_w(vx, vy, wg, par.epsi, par.shape)
    return ft


def compute_F(vx, vy, wg, f_tilde, par):
    # Public compatibility wrapper. It recomputes grad @ wg because the original
    # function signature only receives f_tilde. compute_rhs below avoids this.
    _, gxw, gyw = _compute_ft_and_grad_w(vx, vy, wg, par.epsi, par.shape)
    return _compute_F_second_pass(vx, vy, wg, f_tilde, gxw, gyw, par.epsi, par.shape)


def compute_dv(vx, vy, wg, Fx, Fy, par):
    if par.gam == 0.0:
        return _dv_maxwell(vx, vy, wg, Fx, Fy, par.beta)
    return _dv_direct(vx, vy, wg, Fx, Fy, par.beta, par.gam)


def compute_rhs(vx, vy, wg, par):
    ft, gxw, gyw = _compute_ft_and_grad_w(vx, vy, wg, par.epsi, par.shape)
    Fx, Fy = _compute_F_second_pass(vx, vy, wg, ft, gxw, gyw, par.epsi, par.shape)
    if par.gam == 0.0:
        return _dv_maxwell(vx, vy, wg, Fx, Fy, par.beta)
    return _dv_direct(vx, vy, wg, Fx, Fy, par.beta, par.gam)
