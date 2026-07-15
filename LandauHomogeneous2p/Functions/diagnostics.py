"""Macroscopic quantities and optional conservative rescaling."""
import numpy as np


def compute_moments(vx1, vy1, wg1, vx2, vy2, wg2, par):
    """Return species and total moments for the two-species particle state."""
    n1 = np.sum(wg1)
    n2 = np.sum(wg2)

    U1 = np.array([np.sum(wg1 * vx1) / n1, np.sum(wg1 * vy1) / n1])
    U2 = np.array([np.sum(wg2 * vx2) / n2, np.sum(wg2 * vy2) / n2])

    rho = par.m1 * n1 + par.m2 * n2
    U = (par.m1 * n1 * U1 + par.m2 * n2 * U2) / rho

    T1 = par.m1 * np.sum(wg1 * ((vx1 - U1[0]) ** 2 + (vy1 - U1[1]) ** 2)) / (2.0 * n1)
    T2 = par.m2 * np.sum(wg2 * ((vx2 - U2[0]) ** 2 + (vy2 - U2[1]) ** 2)) / (2.0 * n2)
    T = (
        par.m1 * np.sum(wg1 * ((vx1 - U[0]) ** 2 + (vy1 - U[1]) ** 2))
        + par.m2 * np.sum(wg2 * ((vx2 - U[0]) ** 2 + (vy2 - U[1]) ** 2))
    ) / (2.0 * (n1 + n2))

    total_energy = par.m1 * np.sum(wg1 * (vx1**2 + vy1**2)) + par.m2 * np.sum(wg2 * (vx2**2 + vy2**2))
    total_momentum = np.array([
        par.m1 * np.sum(wg1 * vx1) + par.m2 * np.sum(wg2 * vx2),
        par.m1 * np.sum(wg1 * vy1) + par.m2 * np.sum(wg2 * vy2),
    ])

    return {
        "n1": n1,
        "n2": n2,
        "U1": U1,
        "U2": U2,
        "U": U,
        "T1": T1,
        "T2": T2,
        "T": T,
        "energy": total_energy,
        "momentum": total_momentum,
    }


def conservative_rescale(vx1, vy1, wg1, vx2, vy2, wg2, par, target_temperature):
    """
    MATLAB-style conservative post-collision correction.

    The particles are shifted/scaled around the current total bulk velocity:
        v <- U + sqrt(T_target/T_current) * (v - U).
    This preserves total momentum and sets the total thermal temperature to
    target_temperature.  Since total momentum and masses are unchanged, it also
    restores the corresponding total kinetic energy.
    """
    mom = compute_moments(vx1, vy1, wg1, vx2, vy2, wg2, par)
    current_T = mom["T"]
    if current_T <= 0.0:
        raise ValueError("cannot apply conservative rescaling with non-positive total temperature")

    factor = np.sqrt(target_temperature / current_T)
    Ux, Uy = mom["U"]

    vx1 = Ux + factor * (vx1 - Ux)
    vy1 = Uy + factor * (vy1 - Uy)
    vx2 = Ux + factor * (vx2 - Ux)
    vy2 = Uy + factor * (vy2 - Uy)
    return vx1, vy1, vx2, vy2


def discrete_entropy(vx, vy, wg, epsi, shape, block_size, psi_grad_func):
    """
    Particle-level regularized entropy evaluated at the particle locations.

    This is a diagnostic, not needed by the time integrator.  The argument
    psi_grad_func must be a callable returning (psi, gx, gy), as in kernels.py.
    """
    n = vx.shape[0]
    entropy = 0.0
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        dx = vx[start:end, None] - vx[None, :]
        dy = vy[start:end, None] - vy[None, :]
        psi, _, _ = psi_grad_func(dx, dy, epsi)
        ft = psi @ wg
        entropy += np.sum(wg[start:end] * np.log(np.maximum(ft, np.finfo(float).tiny)))
    return entropy
