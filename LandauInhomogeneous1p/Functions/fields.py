"""1D-2V PIC current deposition, field interpolation, and Yee/Ampere update."""
import numpy as np
from numba import njit, prange

from .splines_numba import spline_value_derivative, periodic_delta


@njit(fastmath=True, cache=True)
def deposit_moments_to_primal_grid(x, v1, v2, w, Lx, eta, Nx, order, radius):
    """
    Deposit density and current to primal grid centres x_i=(i+1/2)*eta.

    rho_i = sum_p w_p psi_eta(x_i-x_p),
    J_i   = sum_p w_p v_p psi_eta(x_i-x_p).
    """
    rho = np.zeros(Nx, dtype=np.float64)
    J1 = np.zeros(Nx, dtype=np.float64)
    J2 = np.zeros(Nx, dtype=np.float64)
    r = int(np.ceil(radius)) + 2

    for p in range(x.shape[0]):
        # nearest lower grid-centre index in the shifted coordinate x/eta - 1/2
        base = int(np.floor(x[p] / eta - 0.5))
        for off in range(-r, r + 1):
            ii = base + off
            idx = ii % Nx
            xg = (idx + 0.5) * eta
            dx = periodic_delta(xg, x[p], Lx)
            b, _ = spline_value_derivative(dx / eta, order)
            if b != 0.0:
                val = w[p] * b / eta
                rho[idx] += val
                J1[idx] += val * v1[p]
                J2[idx] += val * v2[p]
    return rho, J1, J2


@njit(parallel=True, fastmath=True, cache=True)
def interpolate_primal_to_particles(field, x, Lx, eta, Nx, order, radius):
    """Interpolate a primal-centred scalar field to particles with unnormalised B-spline weights."""
    out = np.empty(x.shape[0], dtype=np.float64)
    r = int(np.ceil(radius)) + 2
    for p in prange(x.shape[0]):
        base = int(np.floor(x[p] / eta - 0.5))
        s = 0.0
        for off in range(-r, r + 1):
            ii = base + off
            idx = ii % Nx
            xg = (idx + 0.5) * eta
            dx = periodic_delta(x[p], xg, Lx)
            b, _ = spline_value_derivative(dx / eta, order)
            if b != 0.0:
                s += field[idx] * b
        out[p] = s
    return out


@njit(parallel=True, fastmath=True, cache=True)
def interpolate_dual_to_particles(field, x, Lx, eta, Nx, order, radius):
    """Interpolate a dual/face-centred scalar field at x_i=i*eta to particles."""
    out = np.empty(x.shape[0], dtype=np.float64)
    r = int(np.ceil(radius)) + 2
    for p in prange(x.shape[0]):
        base = int(np.floor(x[p] / eta))
        s = 0.0
        for off in range(-r, r + 1):
            ii = base + off
            idx = ii % Nx
            xg = idx * eta
            dx = periodic_delta(x[p], xg, Lx)
            b, _ = spline_value_derivative(dx / eta, order)
            if b != 0.0:
                s += field[idx] * b
        out[p] = s
    return out


def solve_periodic_poisson_E_from_density(rho, par):
    """
    Solve dE/dx = rho-rho_ion with periodic boundary conditions and zero mean E.
    Uses FFT on the primal grid.
    """
    rhs = rho - par.rho_ion
    rhs = rhs - np.mean(rhs)
    rhs_hat = np.fft.fft(rhs)
    wave = 2.0 * np.pi * np.fft.fftfreq(par.Nx, d=par.eta)
    E_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    mask = wave != 0.0
    E_hat[mask] = rhs_hat[mask] / (1j * wave[mask])
    E = np.fft.ifft(E_hat).real
    E -= np.mean(E)
    return E


def initialize_fields(x, v1, v2, w, par):
    """Initialise E1,E2 on the primal grid and B3 on the dual grid."""
    if par.initial_field == "analytic_landau":
        E1 = (par.alpha / par.k) * np.sin(par.k * par.x_grid)
    else:
        rho, _, _ = deposit_moments_to_primal_grid(
            x, v1, v2, w, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius
        )
        E1 = solve_periodic_poisson_E_from_density(rho, par)

    E2 = np.zeros(par.Nx, dtype=float)
    B3 = np.zeros(par.Nx, dtype=float)
    return E1, E2, B3


def advance_fields(E1, E2, B3, J1, J2, par):
    """
    Advance fields by one explicit step.

    field_solver='ampere': E1_t=-J1, E2=B3=0.
    field_solver='yee': 1D-2V Yee update from section 2.6.1.
    """
    if par.field_solver == "ampere":
        E1n = E1 - par.dt * J1
        E1n -= np.mean(E1n)  # remove any tiny incompatible mean-current drift
        E2n = np.zeros_like(E2)
        B3n = np.zeros_like(B3)
        return E1n, E2n, B3n

    # Yee lattice: E1,E2 on primal centres, B3 on dual faces.
    E1n = E1 - par.dt * J1
    curlB = (np.roll(B3, -1) - B3) / par.eta
    E2n = E2 - par.dt * J2 - par.dt * curlB
    dE2 = (E2 - np.roll(E2, 1)) / par.eta
    B3n = B3 - par.dt * dE2
    return E1n, E2n, B3n


def fields_at_particles(E1, E2, B3, x, par):
    """Interpolate grid fields to particle positions."""
    Ep1 = interpolate_primal_to_particles(E1, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    if par.field_solver == "ampere":
        return Ep1, np.zeros_like(Ep1), np.zeros_like(Ep1)
    Ep2 = interpolate_primal_to_particles(E2, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    Bp3 = interpolate_dual_to_particles(B3, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    return Ep1, Ep2, Bp3
