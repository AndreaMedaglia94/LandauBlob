"""
Reconstructs the blob solution f_tilde(v) = sum_i w_i psi_eps(v - v_i)
on an arbitrary set of evaluation points (typically the mesh centers),
for comparison against the exact solution. Direct translation of
f_recon.m, unified across all four mollifier shapes via PSI_KERNELS
(only psi is ever needed here, so the psi-only kernel is used -- see
physics.py's docstring for why that matters).

Blocked over grid points for the same reason as physics.py: an
(N_particles x N_grid) array can blow up peak memory just as easily as
an (N x N) one.
"""
import numpy as np
from .kernels import PSI_KERNELS, pairwise_diffs


def reconstruct(vx, vy, wg, grid_x, grid_y, par):
    kernel = PSI_KERNELS[par.shape]
    N_grid = grid_x.shape[0]
    f = np.empty(N_grid)
    for start in range(0, N_grid, par.block_size):
        end = min(start + par.block_size, N_grid)
        dx, dy = pairwise_diffs(vx, vy, grid_x[start:end], grid_y[start:end])
        psi = kernel(dx, dy, par.epsi)
        f[start:end] = wg @ psi
    return f