"""1D-2V multispecies PIC charge/current deposition and field updates."""
import numpy as np
from numba import njit, prange

from .splines_numba import spline_value_derivative, periodic_delta


@njit(inline="always", fastmath=True)
def _hat_value(u):
    a = u if u >= 0.0 else -u
    if a <= 1.0:
        return 1.0 - a
    return 0.0


@njit(fastmath=True, cache=True)
def _deposit_one_species(x, v1, v2, w, charge_factor, current_factor,
                         Lx, eta, Nx, order, radius, rho, J1, J2):
    r = int(np.ceil(radius)) + 2
    for p in range(x.shape[0]):
        base = int(np.floor(x[p] / eta - 0.5))
        for off in range(-r, r + 1):
            ii = base + off
            idx = ii % Nx
            xg = (idx + 0.5) * eta
            dx = periodic_delta(xg, x[p], Lx)
            b, _ = spline_value_derivative(dx / eta, order)
            if b != 0.0:
                number_density = w[p] * b / eta
                rho[idx] += charge_factor * number_density
                J1[idx] += current_factor * number_density * v1[p]
                J2[idx] += current_factor * number_density * v2[p]


@njit(fastmath=True, cache=True)
def _deposit_one_species_hat(x, v1, v2, w, charge_factor, current_factor,
                             Lx, eta, Nx, rho, J1, J2):
    inv_eta = 1.0 / eta
    for p in range(x.shape[0]):
        base = int(np.floor(x[p] * inv_eta - 0.5))
        for off in range(2):
            ii = base + off
            idx = ii % Nx
            xg = (idx + 0.5) * eta
            dx = periodic_delta(xg, x[p], Lx)
            b = _hat_value(dx * inv_eta)
            if b != 0.0:
                number_density = w[p] * b * inv_eta
                rho[idx] += charge_factor * number_density
                J1[idx] += current_factor * number_density * v1[p]
                J2[idx] += current_factor * number_density * v2[p]


@njit(fastmath=True, cache=True)
def deposit_charge_current_to_primal_grid(x1, v11, v12, w1, x2, v21, v22, w2,
                                          charge1, charge2, current_factor1, current_factor2,
                                          Lx, eta, Nx, order, radius, rho_background):
    rho = np.zeros(Nx, dtype=np.float64)
    J1 = np.zeros(Nx, dtype=np.float64)
    J2 = np.zeros(Nx, dtype=np.float64)
    if order == 1:
        _deposit_one_species_hat(x1, v11, v12, w1, charge1, current_factor1, Lx, eta, Nx, rho, J1, J2)
        _deposit_one_species_hat(x2, v21, v22, w2, charge2, current_factor2, Lx, eta, Nx, rho, J1, J2)
    else:
        _deposit_one_species(x1, v11, v12, w1, charge1, current_factor1, Lx, eta, Nx, order, radius, rho, J1, J2)
        _deposit_one_species(x2, v21, v22, w2, charge2, current_factor2, Lx, eta, Nx, order, radius, rho, J1, J2)
    if rho_background != 0.0:
        for i in range(Nx):
            rho[i] += rho_background
    return rho, J1, J2


@njit(parallel=True, fastmath=True, cache=True)
def interpolate_primal_to_particles(field, x, Lx, eta, Nx, order, radius):
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


@njit(parallel=True, fastmath=True, cache=True)
def interpolate_primal_to_particles_hat(field, x, Lx, eta, Nx):
    out = np.empty(x.shape[0], dtype=np.float64)
    inv_eta = 1.0 / eta
    for p in prange(x.shape[0]):
        base = int(np.floor(x[p] * inv_eta - 0.5))
        s = 0.0
        for off in range(2):
            ii = base + off
            idx = ii % Nx
            xg = (idx + 0.5) * eta
            dx = periodic_delta(x[p], xg, Lx)
            b = _hat_value(dx * inv_eta)
            if b != 0.0:
                s += field[idx] * b
        out[p] = s
    return out


@njit(parallel=True, fastmath=True, cache=True)
def interpolate_dual_to_particles_hat(field, x, Lx, eta, Nx):
    out = np.empty(x.shape[0], dtype=np.float64)
    inv_eta = 1.0 / eta
    for p in prange(x.shape[0]):
        base = int(np.floor(x[p] * inv_eta))
        s = 0.0
        for off in range(2):
            ii = base + off
            idx = ii % Nx
            xg = idx * eta
            dx = periodic_delta(x[p], xg, Lx)
            b = _hat_value(dx * inv_eta)
            if b != 0.0:
                s += field[idx] * b
        out[p] = s
    return out


def solve_periodic_poisson_E_from_charge_density(rho, par):
    """Solve dE/dx=rho with periodic BC and zero mean E on primal centres."""
    rhs = rho - np.mean(rho)
    rhs_hat = np.fft.fft(rhs)
    wave = 2.0 * np.pi * np.fft.fftfreq(par.Nx, d=par.eta)
    E_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    mask = wave != 0.0
    E_hat[mask] = rhs_hat[mask] / (1j * wave[mask])
    E = np.fft.ifft(E_hat).real
    E -= np.mean(E)
    return E


def initialize_fields(x1, v11, v12, w1, x2, v21, v22, w2, par):
    """Initialise E1,E2 on primal centres and B3 on dual faces."""
    if par.initial_field == "analytic_landau":
        # Poisson: dE/dx = rho.  Only the cosine perturbations contribute.
        coeff = par.charge1 * par.n1 * par.alpha1 + par.charge2 * par.n2 * par.alpha2
        E1 = (coeff / par.k) * np.sin(par.k * par.x_grid)
        if par.remove_mean_E1:
            E1 -= np.mean(E1)
    else:
        rho, _, _ = deposit_charge_current_to_primal_grid(
            x1, v11, v12, w1, x2, v21, v22, w2,
            par.charge1, par.charge2, par.current_factor1, par.current_factor2,
            par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, par.rho_background
        )
        E1 = solve_periodic_poisson_E_from_charge_density(rho, par)

    E2 = np.zeros(par.Nx, dtype=float)
    B3 = np.zeros(par.Nx, dtype=float)
    return E1, E2, B3


def advance_fields(E1, E2, B3, J1, J2, par):
    """Advance fields by one explicit Ampere or Yee step."""
    if par.field_solver == "ampere":
        E1n = E1 - par.dt * J1
        if par.remove_mean_E1:
            E1n -= np.mean(E1n)
        E2n = np.zeros_like(E2)
        B3n = np.zeros_like(B3)
        return E1n, E2n, B3n

    E1n = E1 - par.dt * J1
    if par.remove_mean_E1:
        E1n -= np.mean(E1n)
    curlB = (np.roll(B3, -1) - B3) / par.eta
    E2n = E2 - par.dt * J2 - par.dt * curlB
    dE2 = (E2 - np.roll(E2, 1)) / par.eta
    B3n = B3 - par.dt * dE2
    return E1n, E2n, B3n


def fields_at_particles(E1, E2, B3, x, par):
    """Interpolate grid fields to one species' particle positions."""
    if par.spline_order == 1:
        Ep1 = interpolate_primal_to_particles_hat(E1, x, par.Lx, par.eta, par.Nx)
        if par.field_solver == "ampere":
            z = np.zeros_like(Ep1)
            return Ep1, z, z.copy()
        Ep2 = interpolate_primal_to_particles_hat(E2, x, par.Lx, par.eta, par.Nx)
        Bp3 = interpolate_dual_to_particles_hat(B3, x, par.Lx, par.eta, par.Nx)
        return Ep1, Ep2, Bp3

    Ep1 = interpolate_primal_to_particles(E1, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    if par.field_solver == "ampere":
        z = np.zeros_like(Ep1)
        return Ep1, z, z.copy()
    Ep2 = interpolate_primal_to_particles(E2, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    Bp3 = interpolate_dual_to_particles(B3, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    return Ep1, Ep2, Bp3
