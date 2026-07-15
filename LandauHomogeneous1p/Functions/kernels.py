"""
Mollifiers psi_epsilon and their gradients, evaluated pairwise between
two point sets.

Two interfaces are provided:

  KERNELS[shape](dx, dy, epsi) -> (psi, grad_x, grad_y)
      The original combined form. Used by physics_dense.py (the
      correctness reference) and anywhere both outputs are needed.

  PSI_KERNELS[shape](dx, dy, epsi)  -> psi only
  GRAD_KERNELS[shape](dx, dy, epsi) -> (grad_x, grad_y) only
      Specialized forms that skip computing the output nobody asked
      for. compute_f_tilde() and reconstruct() only ever use psi;
      compute_F() only ever uses the gradient. Benchmarked at ~2x
      faster than the combined form for exactly this reason (this
      code is memory-bandwidth-bound, so not allocating/writing two
      full (block, N) arrays that would be immediately discarded
      matters far more than the FLOPs themselves) -- see chat for the
      measurements.

All three dicts are numerically IDENTICAL where they overlap; this is
verified by regression test against physics_dense.py across every
shape and every gamma (see test_regression.py).

Note: Gaussian's epsi is a *variance* (matches par.epsi = 0.64*h^1.98
in the paper's regularization), while the B-splines' epsi is a
*bandwidth/support radius* (par.epsi = h) -- these are genuinely
different quantities, exactly as in the MATLAB code, not a translation
inconsistency.
"""
import numpy as np


def pairwise_diffs(vx1, vy1, vx2, vy2):
    """dx[i,j] = vx1[i] - vx2[j], dy[i,j] = vy1[i] - vy2[j]."""
    dx = vx1[:, None] - vx2[None, :]
    dy = vy1[:, None] - vy2[None, :]
    return dx, dy


# ---------------------------------------------------------------- Gaussian
def gaussian(dx, dy, epsi):
    psi = np.exp(-(dx**2 + dy**2) / (2 * epsi)) / (2 * np.pi * epsi)
    grad_x = -dx / epsi * psi
    grad_y = -dy / epsi * psi
    return psi, grad_x, grad_y


def gaussian_psi(dx, dy, epsi):
    return np.exp(-(dx**2 + dy**2) / (2 * epsi)) / (2 * np.pi * epsi)


def gaussian_grad(dx, dy, epsi):
    psi = np.exp(-(dx**2 + dy**2) / (2 * epsi)) / (2 * np.pi * epsi)  # unavoidable intermediate
    return -dx / epsi * psi, -dy / epsi * psi


# ---------------------------------------------------------------- B-spline 1
def bspline1(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    Gx = (1 - aX) * (aX <= 1)
    Gy = (1 - aY) * (aY <= 1)
    psi = Gx * Gy / epsi**2
    grad_x = Gy / epsi**3 * (-np.sign(X) * (aX <= 1))
    grad_y = Gx / epsi**3 * (-np.sign(Y) * (aY <= 1))
    return psi, grad_x, grad_y


def bspline1_psi(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    Gx = (1 - aX) * (aX <= 1)
    Gy = (1 - aY) * (aY <= 1)
    return Gx * Gy / epsi**2


def bspline1_grad(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    Gx = (1 - aX) * (aX <= 1)
    Gy = (1 - aY) * (aY <= 1)
    grad_x = Gy / epsi**3 * (-np.sign(X) * (aX <= 1))
    grad_y = Gx / epsi**3 * (-np.sign(Y) * (aY <= 1))
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

    dGx = -2 * X * inner_x - np.sign(X) * (1.5 - aX) * outer_x
    dGy = -2 * Y * inner_y - np.sign(Y) * (1.5 - aY) * outer_y
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
    dGx = -2 * X * inner_x - np.sign(X) * (1.5 - aX) * outer_x
    dGy = -2 * Y * inner_y - np.sign(Y) * (1.5 - aY) * outer_y
    grad_x = Gy / epsi**3 * dGx
    grad_y = Gx / epsi**3 * dGy
    return grad_x, grad_y


# ---------------------------------------------------------------- B-spline 3
def bspline3(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 1, (aX > 1) & (aX <= 2)
    inner_y, outer_y = aY <= 1, (aY > 1) & (aY <= 2)

    Gx = (4 - 6 * X**2 + 3 * aX**3) / 6 * inner_x + (2 - aX)**3 / 6 * outer_x
    Gy = (4 - 6 * Y**2 + 3 * aY**3) / 6 * inner_y + (2 - aY)**3 / 6 * outer_y
    psi = Gx * Gy / epsi**2

    dGx = (-12 * X + 9 * X * aX) / 6 * inner_x - 0.5 * (2 - aX)**2 * np.sign(X) * outer_x
    dGy = (-12 * Y + 9 * Y * aY) / 6 * inner_y - 0.5 * (2 - aY)**2 * np.sign(Y) * outer_y
    grad_x = Gy / epsi**3 * dGx
    grad_y = Gx / epsi**3 * dGy
    return psi, grad_x, grad_y


def bspline3_psi(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 1, (aX > 1) & (aX <= 2)
    inner_y, outer_y = aY <= 1, (aY > 1) & (aY <= 2)
    Gx = (4 - 6 * X**2 + 3 * aX**3) / 6 * inner_x + (2 - aX)**3 / 6 * outer_x
    Gy = (4 - 6 * Y**2 + 3 * aY**3) / 6 * inner_y + (2 - aY)**3 / 6 * outer_y
    return Gx * Gy / epsi**2


def bspline3_grad(dx, dy, epsi):
    X, Y = dx / epsi, dy / epsi
    aX, aY = np.abs(X), np.abs(Y)
    inner_x, outer_x = aX <= 1, (aX > 1) & (aX <= 2)
    inner_y, outer_y = aY <= 1, (aY > 1) & (aY <= 2)
    Gx = (4 - 6 * X**2 + 3 * aX**3) / 6 * inner_x + (2 - aX)**3 / 6 * outer_x
    Gy = (4 - 6 * Y**2 + 3 * aY**3) / 6 * inner_y + (2 - aY)**3 / 6 * outer_y
    dGx = (-12 * X + 9 * X * aX) / 6 * inner_x - 0.5 * (2 - aX)**2 * np.sign(X) * outer_x
    dGy = (-12 * Y + 9 * Y * aY) / 6 * inner_y - 0.5 * (2 - aY)**2 * np.sign(Y) * outer_y
    grad_x = Gy / epsi**3 * dGx
    grad_y = Gx / epsi**3 * dGy
    return grad_x, grad_y


KERNELS = {0: gaussian, 1: bspline1, 2: bspline2, 3: bspline3}
PSI_KERNELS = {0: gaussian_psi, 1: bspline1_psi, 2: bspline2_psi, 3: bspline3_psi}
GRAD_KERNELS = {0: gaussian_grad, 1: bspline1_grad, 2: bspline2_grad, 3: bspline3_grad}