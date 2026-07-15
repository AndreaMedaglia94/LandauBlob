"""
Blocked (chunked) evaluation of the particle RHS.

The dense version (physics_dense.py, kept as a correctness reference)
materializes full NxN arrays -- psi, grad_psi, A11, A12, A22, all of it
at once. That's what made the MATLAB translation fast via BLAS
matrix-vector products, but it also means peak memory grows as O(N^2):
doubling N quadruples memory. Measured on this machine, N=10,000 in 2D
already gets OOM-killed, well before FLOP count alone would suggest a
problem.

The fix here is NOT an algorithmic change -- this is still the exact
same O(N^2) direct sum (no treecode, no far-field approximation,
numerically identical to physics_dense.py). It only changes *how much
of the NxN structure exists in memory at once*: particles are processed
in row-blocks of size par.block_size, so peak memory is
O(block_size * N) instead of O(N^2), and each block's inner computation
is still a single BLAS matrix-vector product ("@"), so per-flop speed
is essentially unchanged -- we've bounded the peak, not slowed the math.

Derivation of the blocked velocity update:

    dv_i/dt = -sum_j w_j A(v_i-v_j) [F(v_i) - F(v_j)]

Expanding the x-component (A21 = A12 by symmetry):

    dvx_i = -Fx_i * R11_i + T11_i - Fy_i * R12_i + T12y_i
    dvy_i = -Fx_i * R12_i + T12x_i - Fy_i * R22_i + T22_i

where, for a row-block of receivers i against ALL sources j:

    R11_i = sum_j w_j A11_ij      = (A11_block @ wg)_i
    T11_i = sum_j w_j A11_ij Fx_j = (A11_block @ (wg*Fx))_i
    ... and likewise for R12, R22, T12x, T12y, T22.

Each of these is a matrix-vector product against a block of A, which is
discarded immediately after -- it never needs to exist for the full N.

This block-over-receivers loop is also the natural seed for later
batching over spatial cells: a "block of particles" today becomes
"particles in one cell" once you go inhomogeneous, and the same
per-block matrix-vector pattern is exactly what vmaps cleanly onto a
GPU with JAX/PyTorch when that's worth doing.
"""
import numpy as np
from .kernels import KERNELS, pairwise_diffs


def _blocks(n, block_size):
    for start in range(0, n, block_size):
        yield slice(start, min(start + block_size, n))


def compute_f_tilde(vx, vy, wg, par):
    """f_tilde[i] = sum_k w_k psi(v_i - v_k). O(N) memory for the result;
    O(block_size * N) peak memory while computing it."""
    kernel = KERNELS[par.shape]
    N = vx.shape[0]
    f_tilde = np.empty(N)
    for blk in _blocks(N, par.block_size):
        dx, dy = pairwise_diffs(vx[blk], vy[blk], vx, vy)
        psi, _, _ = kernel(dx, dy, par.epsi)
        f_tilde[blk] = psi @ wg
    return f_tilde


def compute_F(vx, vy, wg, f_tilde, par):
    """F(v_i), the regularized-entropy gradient (eq. 3.9), per particle."""
    kernel = KERNELS[par.shape]
    N = vx.shape[0]
    Fx, Fy = np.empty(N), np.empty(N)
    w_over_ft = wg / f_tilde  # global N-vector, reused every block
    for blk in _blocks(N, par.block_size):
        dx, dy = pairwise_diffs(vx[blk], vy[blk], vx, vy)
        _, gx, gy = kernel(dx, dy, par.epsi)
        Fx[blk] = (gx @ wg) / f_tilde[blk] + gx @ w_over_ft
        Fy[blk] = (gy @ wg) / f_tilde[blk] + gy @ w_over_ft
    return Fx, Fy


def compute_dv(vx, vy, wg, Fx, Fy, par):
    """dv/dt for every particle, via the blocked expansion derived above."""
    N = vx.shape[0]
    dvx, dvy = np.empty(N), np.empty(N)
    wFx, wFy = wg * Fx, wg * Fy
    for blk in _blocks(N, par.block_size):
        dx, dy = pairwise_diffs(vx[blk], vy[blk], vx, vy)
        r2 = dx**2 + dy**2
        # r2>0 masking is safe for any sign of gam (see physics_dense.py's
        # get_A_dense docstring); errstate silences the benign warning from
        # np.where evaluating 0**negative on the branch it then discards
        with np.errstate(divide="ignore", invalid="ignore"):
            nn = np.where(r2 > 0, par.beta * r2 ** (par.gam / 2), 0.0)
        A11 = nn * dy**2
        A12 = -nn * dx * dy
        A22 = nn * dx**2

        R11, R12, R22 = A11 @ wg, A12 @ wg, A22 @ wg
        T11, T12x = A11 @ wFx, A12 @ wFx
        T12y, T22 = A12 @ wFy, A22 @ wFy

        dvx[blk] = (T11 - Fx[blk] * R11) + (T12y - Fy[blk] * R12)
        dvy[blk] = (T12x - Fx[blk] * R12) + (T22 - Fy[blk] * R22)
    return dvx, dvy


def compute_rhs(vx, vy, wg, par):
    """Full RHS dv/dt of the particle ODE system, in bounded memory."""
    f_tilde = compute_f_tilde(vx, vy, wg, par)
    Fx, Fy = compute_F(vx, vy, wg, f_tilde, par)
    return compute_dv(vx, vy, wg, Fx, Fy, par)
