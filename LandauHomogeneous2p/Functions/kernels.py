"""
Mollifiers psi_epsilon and gradients used by the homogeneous Landau
particle code.

shape = 0: Gaussian, where epsi is the variance.
shape = 1,2,3: tensor-product B-splines, where epsi is the bandwidth.
"""
import numpy as np


def pairwise_diffs(vx1, vy1, vx2, vy2):
    """dx[i,j] = vx1[i] - vx2[j], dy[i,j] = vy1[i] - vy2[j]."""
    dx = vx1[:, None] - vx2[None, :]
    dy = vy1[:, None] - vy2[None, :]
    return dx, dy


# ---------------------------------------------------------------- Gaussian
def gaussian(dx, dy, epsi):
    psi = np.exp(-(dx**2 + dy**2) / (2.0 * epsi)) / (2.0 * np.pi * epsi)
    grad_x = -dx / epsi * psi
    grad_y = -dy / epsi * psi
    return psi, grad_x, grad_y


def gaussian_psi(dx, dy, epsi):
    return np.exp(-(dx**2 + dy**2) / (2.0 * epsi)) / (2.0 * np.pi * epsi)


def gaussian_grad(dx, dy, epsi):
    psi = gaussian_psi(dx, dy, epsi)
    return -dx / epsi * psi, -dy / epsi * psi


# ---------------------------------------------------------------- B-spline 1
def bspline1(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    Gx = (1.0 - aX) * (aX <= 1.0)
    Gy = (1.0 - aY) * (aY <= 1.0)
    psi = Gx * Gy / epsi**2
    grad_x = Gy / epsi**3 * (-np.sign(X) * (aX <= 1.0))
    grad_y = Gx / epsi**3 * (-np.sign(Y) * (aY <= 1.0))
    return psi, grad_x, grad_y


def bspline1_psi(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    Gx = (1.0 - aX) * (aX <= 1.0)
    Gy = (1.0 - aY) * (aY <= 1.0)
    return Gx * Gy / epsi**2


def bspline1_grad(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    Gx = (1.0 - aX) * (aX <= 1.0)
    Gy = (1.0 - aY) * (aY <= 1.0)
    grad_x = Gy / epsi**3 * (-np.sign(X) * (aX <= 1.0))
    grad_y = Gx / epsi**3 * (-np.sign(Y) * (aY <= 1.0))
    return grad_x, grad_y


# ---------------------------------------------------------------- B-spline 2
def bspline2(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 0.5, (aX > 0.5) & (aX <= 1.5)
    inner_y, outer_y = aY <= 0.5, (aY > 0.5) & (aY <= 1.5)

    Gx = (0.75 - X**2) * inner_x + 0.5 * (1.5 - aX)**2 * outer_x
    Gy = (0.75 - Y**2) * inner_y + 0.5 * (1.5 - aY)**2 * outer_y
    psi = Gx * Gy / epsi**2

    dGx = -2.0 * X * inner_x - np.sign(X) * (1.5 - aX) * outer_x
    dGy = -2.0 * Y * inner_y - np.sign(Y) * (1.5 - aY) * outer_y
    grad_x = Gy / epsi**3 * dGx
    grad_y = Gx / epsi**3 * dGy
    return psi, grad_x, grad_y


def bspline2_psi(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 0.5, (aX > 0.5) & (aX <= 1.5)
    inner_y, outer_y = aY <= 0.5, (aY > 0.5) & (aY <= 1.5)
    Gx = (0.75 - X**2) * inner_x + 0.5 * (1.5 - aX)**2 * outer_x
    Gy = (0.75 - Y**2) * inner_y + 0.5 * (1.5 - aY)**2 * outer_y
    return Gx * Gy / epsi**2


def bspline2_grad(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 0.5, (aX > 0.5) & (aX <= 1.5)
    inner_y, outer_y = aY <= 0.5, (aY > 0.5) & (aY <= 1.5)
    Gx = (0.75 - X**2) * inner_x + 0.5 * (1.5 - aX)**2 * outer_x
    Gy = (0.75 - Y**2) * inner_y + 0.5 * (1.5 - aY)**2 * outer_y
    dGx = -2.0 * X * inner_x - np.sign(X) * (1.5 - aX) * outer_x
    dGy = -2.0 * Y * inner_y - np.sign(Y) * (1.5 - aY) * outer_y
    grad_x = Gy / epsi**3 * dGx
    grad_y = Gx / epsi**3 * dGy
    return grad_x, grad_y


# ---------------------------------------------------------------- B-spline 3
def bspline3(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 1.0, (aX > 1.0) & (aX <= 2.0)
    inner_y, outer_y = aY <= 1.0, (aY > 1.0) & (aY <= 2.0)

    Gx = (4.0 - 6.0 * X**2 + 3.0 * aX**3) / 6.0 * inner_x + (2.0 - aX)**3 / 6.0 * outer_x
    Gy = (4.0 - 6.0 * Y**2 + 3.0 * aY**3) / 6.0 * inner_y + (2.0 - aY)**3 / 6.0 * outer_y
    psi = Gx * Gy / epsi**2

    dGx = (-12.0 * X + 9.0 * X * aX) / 6.0 * inner_x - 0.5 * (2.0 - aX)**2 * np.sign(X) * outer_x
    dGy = (-12.0 * Y + 9.0 * Y * aY) / 6.0 * inner_y - 0.5 * (2.0 - aY)**2 * np.sign(Y) * outer_y
    grad_x = Gy / epsi**3 * dGx
    grad_y = Gx / epsi**3 * dGy
    return psi, grad_x, grad_y


def bspline3_psi(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 1.0, (aX > 1.0) & (aX <= 2.0)
    inner_y, outer_y = aY <= 1.0, (aY > 1.0) & (aY <= 2.0)
    Gx = (4.0 - 6.0 * X**2 + 3.0 * aX**3) / 6.0 * inner_x + (2.0 - aX)**3 / 6.0 * outer_x
    Gy = (4.0 - 6.0 * Y**2 + 3.0 * aY**3) / 6.0 * inner_y + (2.0 - aY)**3 / 6.0 * outer_y
    return Gx * Gy / epsi**2


def bspline3_grad(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 1.0, (aX > 1.0) & (aX <= 2.0)
    inner_y, outer_y = aY <= 1.0, (aY > 1.0) & (aY <= 2.0)
    Gx = (4.0 - 6.0 * X**2 + 3.0 * aX**3) / 6.0 * inner_x + (2.0 - aX)**3 / 6.0 * outer_x
    Gy = (4.0 - 6.0 * Y**2 + 3.0 * aY**3) / 6.0 * inner_y + (2.0 - aY)**3 / 6.0 * outer_y
    dGx = (-12.0 * X + 9.0 * X * aX) / 6.0 * inner_x - 0.5 * (2.0 - aX)**2 * np.sign(X) * outer_x
    dGy = (-12.0 * Y + 9.0 * Y * aY) / 6.0 * inner_y - 0.5 * (2.0 - aY)**2 * np.sign(Y) * outer_y
    grad_x = Gy / epsi**3 * dGx
    grad_y = Gx / epsi**3 * dGy
    return grad_x, grad_y


KERNELS = {0: gaussian, 1: bspline1, 2: bspline2, 3: bspline3}
PSI_KERNELS = {0: gaussian_psi, 1: bspline1_psi, 2: bspline2_psi, 3: bspline3_psi}
GRAD_KERNELS = {0: gaussian_grad, 1: bspline1_grad, 2: bspline2_grad, 3: bspline3_grad}
