"""Blob reconstruction f_tilde(v) = sum_p w_p psi_epsilon(v - v_p)."""
import numpy as np
from .kernels import PSI_KERNELS, pairwise_diffs


def reconstruct(vx, vy, wg, grid_x, grid_y, shape, epsi, block_size=1000):
    """Evaluate the regularized particle density on an arbitrary grid."""
    kernel = PSI_KERNELS[shape]
    n_grid = grid_x.shape[0]
    f = np.empty(n_grid)
    for start in range(0, n_grid, block_size):
        end = min(start + block_size, n_grid)
        dx, dy = pairwise_diffs(vx, vy, grid_x[start:end], grid_y[start:end])
        psi = kernel(dx, dy, epsi)
        f[start:end] = wg @ psi
    return f
