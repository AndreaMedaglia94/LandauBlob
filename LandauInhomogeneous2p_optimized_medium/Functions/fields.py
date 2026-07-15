"""1D-2V multispecies PIC charge/current deposition and field updates.

This is a drop-in replacement for the original fields.py: all public
function signatures and return values are unchanged (deposit_charge_current_
to_primal_grid gained one optional trailing keyword argument, sort_oversample,
with a default that reproduces the original behaviour if omitted).

What changed is purely implementation:

  _deposit_one_species was a serial scatter-add into shared rho/J1/J2
  arrays (each particle can touch several grid points, so a naive prange
  would race on the += into rho/J1/J2). It is replaced by
  _deposit_one_species_chunked: particles are split into `nchunks`
  contiguous chunks, each chunk accumulates into its own private row of a
  (nchunks, Nx) buffer under prange (safe -- chunks own disjoint rows),
  and a final, cheap serial reduction (O(nchunks*Nx), negligible next to
  the O(N*stencil) deposition work) sums the chunks together. This is the
  same "privatised accumulation + reduction" idea used for the collision
  cell-sort's histograms in cell_sort.py.
"""
import numpy as np
from numba import njit, prange

from .splines_numba import spline_value_derivative, periodic_delta
from .cell_sort import default_nchunks


@njit(parallel=True, fastmath=True, cache=True)
def _deposit_one_species_chunked(x, v1, v2, w, charge_factor, current_factor,
                                  Lx, eta, Nx, order, radius, nchunks):
    N = x.shape[0]
    nchunks = max(1, min(nchunks, max(1, N)))
    chunk_size = (N + nchunks - 1) // nchunks
    r = int(np.ceil(radius)) + 2
    inv_eta = 1.0 / eta

    rho_local = np.zeros((nchunks, Nx), dtype=np.float64)
    J1_local = np.zeros((nchunks, Nx), dtype=np.float64)
    J2_local = np.zeros((nchunks, Nx), dtype=np.float64)

    for c in prange(nchunks):
        start = c * chunk_size
        end = start + chunk_size
        if end > N:
            end = N
        for p in range(start, end):
            base = int(np.floor(x[p] * inv_eta - 0.5))
            for off in range(-r, r + 1):
                ii = base + off
                idx = ii % Nx
                xg = (idx + 0.5) * eta
                dx = periodic_delta(xg, x[p], Lx)
                b, _ = spline_value_derivative(dx * inv_eta, order)
                if b != 0.0:
                    number_density = w[p] * b * inv_eta
                    rho_local[c, idx] += charge_factor * number_density
                    J1_local[c, idx] += current_factor * number_density * v1[p]
                    J2_local[c, idx] += current_factor * number_density * v2[p]

    rho = np.zeros(Nx, dtype=np.float64)
    J1 = np.zeros(Nx, dtype=np.float64)
    J2 = np.zeros(Nx, dtype=np.float64)
    for c in range(nchunks):
        for i in range(Nx):
            rho[i] += rho_local[c, i]
            J1[i] += J1_local[c, i]
            J2[i] += J2_local[c, i]
    return rho, J1, J2


def deposit_charge_current_to_primal_grid(x1, v11, v12, w1, x2, v21, v22, w2,
                                          charge1, charge2, current_factor1, current_factor2,
                                          Lx, eta, Nx, order, radius, rho_background,
                                          sort_oversample=1):
    n1chunks = default_nchunks(x1.shape[0], sort_oversample)
    n2chunks = default_nchunks(x2.shape[0], sort_oversample)
    rho1, J11, J21 = _deposit_one_species_chunked(
        x1, v11, v12, w1, charge1, current_factor1, Lx, eta, Nx, order, radius, n1chunks)
    rho2, J12, J22 = _deposit_one_species_chunked(
        x2, v21, v22, w2, charge2, current_factor2, Lx, eta, Nx, order, radius, n2chunks)
    rho = rho1 + rho2
    J1 = J11 + J12
    J2 = J21 + J22
    if rho_background != 0.0:
        rho = rho + rho_background
    return rho, J1, J2


@njit(parallel=True, fastmath=True, cache=True)
def interpolate_primal_to_particles(field, x, Lx, eta, Nx, order, radius):
    out = np.empty(x.shape[0], dtype=np.float64)
    r = int(np.ceil(radius)) + 2
    inv_eta = 1.0 / eta
    for p in prange(x.shape[0]):
        base = int(np.floor(x[p] * inv_eta - 0.5))
        s = 0.0
        for off in range(-r, r + 1):
            ii = base + off
            idx = ii % Nx
            xg = (idx + 0.5) * eta
            dx = periodic_delta(x[p], xg, Lx)
            b, _ = spline_value_derivative(dx * inv_eta, order)
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
            par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, par.rho_background,
            par.sort_oversample
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
    Ep1 = interpolate_primal_to_particles(E1, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    if par.field_solver == "ampere":
        z = np.zeros_like(Ep1)
        return Ep1, z, z.copy()
    Ep2 = interpolate_primal_to_particles(E2, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    Bp3 = interpolate_dual_to_particles(B3, x, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius)
    return Ep1, Ep2, Bp3
