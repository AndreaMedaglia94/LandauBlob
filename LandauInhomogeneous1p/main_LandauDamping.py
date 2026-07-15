"""
One-species, space-inhomogeneous C-PIC code for Landau damping in 1D-2V.

Structure follows the homogeneous Python code:
  1. Params
  2. initialise particles and fields
  3. time cycle
  4. post-processing

Paper-scale Landau damping parameters from section 3.1.2 are:
    Nx=128, Nv1=Nv2=32, Nc=8, T=10, dt=1/50,
    random_batches=32, Lv1=Lv2=4, alpha=0.1, k=0.5,
    gam=-2, collision_strength in {0, 0.01, 0.1, 1}.
This gives N=1,048,576 particles and is not a laptop-size Python test.
Start smaller, then increase resolution.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from Functions.params import Params
from Functions.initialize import initialize_particles_landau_damping, make_batch_ids
from Functions.fields import (
    deposit_moments_to_primal_grid,
    initialize_fields,
    advance_fields,
    fields_at_particles,
)
from Functions.collisions_numba import compute_collision_acceleration, compute_regularized_density_at_particles
from Functions.diagnostics import History
from Functions.progress import ProgressBar


# ----------------------------------------------------------------------
# Edit this block for a run.  The defaults in Params are intentionally
# modest.  Increase Nx,Nv1,Nv2,Nc and random_batches after confirming the
# workflow on your machine.
# ----------------------------------------------------------------------
BASE_KWARGS = dict(
    # Landau damping datum
    alpha=0.1,
    k=0.5,
    Lv1=4.0,
    Lv2=4.0,
    # Resolution.  Paper: Nx=128, Nv1=Nv2=32, Nc=8.
    Nx=128,
    Nv1=32,
    Nv2=32,
    Nc=8,
    # Coulombian 2D velocity kernel.  Paper: gam=-2.
    gam=-2.0,
    collision_strength=0.1,
    # Compact spline order: 1=hat, 2=quadratic, 3=cubic.
    spline_order=1,
    # Time.  Paper: T=10, dt=1/50.
    T=10.0,
    dt=1.0 / 50.0,
    # Landau damping uses Vlasov-Ampere-Landau; use field_solver='yee'
    # for the full 1D-2V staggered Maxwell update.
    field_solver="ampere",
    # Random batch.  Paper: random_batches=32.
    random_batch=True,
    random_batches=32,
    # Output and reproducibility.
    random_seed=12345,
    record_interval=0.1,
    output_prefix="landau_damping",
)
# ----------------------------------------------------------------------


def time_cycle(x, v1, v2, w, E1, E2, B3, par):
    ntot = par.ntot
    steps_per_record = max(1, round(par.record_interval / par.dt))
    rng_batch = np.random.default_rng(None if par.random_seed is None else par.random_seed + 991)

    hist = History()
    # Compute initial entropy once; this is compact-cell-list accelerated.
    ft0 = compute_regularized_density_at_particles(x, v1, v2, w, par) if par.enable_collisions else None
    hist.append(0.0, x, v1, v2, w, E1, E2, B3, par, ft0)

    bar = ProgressBar(ntot, label=f"N={par.N} ")
    start = time.time()

    for step in range(1, ntot + 1):
        # Collision force at f^n.  Steps I-II use phase-space cell lists;
        # Step III uses a spatial cell list and optional random batches.
        if par.enable_collisions and par.collision_strength != 0.0:
            batch_ids = make_batch_ids(par.N, par, rng_batch)
            a_col1, a_col2, ft, _, _ = compute_collision_acceleration(x, v1, v2, w, batch_ids, par)
        else:
            a_col1 = np.zeros_like(v1)
            a_col2 = np.zeros_like(v2)
            ft = None

        # PIC deposition and field update on the staggered 1D grid.
        _, J1, J2 = deposit_moments_to_primal_grid(
            x, v1, v2, w, par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius
        )
        E1, E2, B3 = advance_fields(E1, E2, B3, J1, J2, par)
        Ep1, Ep2, Bp3 = fields_at_particles(E1, E2, B3, x, par)

        # Particle update, matching the 1D-2V Lorentz structure:
        # dv1/dt = E1 + v2*B3 - U1, dv2/dt = E2 - v1*B3 - U2.
        old_v1 = v1.copy()
        v1 = v1 + par.dt * (Ep1 + v2 * Bp3 + a_col1)
        v2 = v2 + par.dt * (Ep2 - old_v1 * Bp3 + a_col2)
        x = (x + par.dt * v1) % par.Lx

        if step % steps_per_record == 0 or step == ntot:
            # The ft computed above corresponds to f^n, not exactly f^{n+1};
            # for inexpensive diagnostics we reuse it.  For exact diagnostic
            # entropy at the record time, uncomment the next line.
            # ft = compute_regularized_density_at_particles(x, v1, v2, w, par)
            hist.append(step * par.dt, x, v1, v2, w, E1, E2, B3, par, ft)

        bar.update(step)

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.2f} s")
    return x, v1, v2, w, E1, E2, B3, hist


def save_outputs(x, v1, v2, w, E1, E2, B3, hist, par):
    os.makedirs("LandauInhomogeneous1p/Results", exist_ok=True)
    os.makedirs("LandauInhomogeneous1p/Figures", exist_ok=True)
    data = hist.as_dict()
    if par.save_results:
        path = os.path.join("LandauInhomogeneous1p/Results", f"{par.output_prefix}.npz")
        np.savez(
            path,
            x=x,
            v1=v1,
            v2=v2,
            w=w,
            E1=E1,
            E2=E2,
            B3=B3,
            **data,
        )
        print(f"Saved results to {path}")

    if par.make_plots:
        t = data["t"]
        E = np.maximum(data["E1_l2"], 1.0e-300)
        plt.figure()
        plt.semilogy(t, E, marker="o", label="||E1||_2")
        if E.size > 0:
            ref0 = E[0] * np.exp(par.gamma_landau_collisionless * t)
            refc = E[0] * np.exp((par.gamma_landau_collisionless + par.collision_strength * par.gamma_landau_collisional_correction) * t)
            plt.semilogy(t, ref0, linestyle="--", label="collisionless envelope")
            if par.collision_strength != 0.0:
                plt.semilogy(t, refc, linestyle=":", label="weak-collision envelope")
        plt.xlabel("time")
        plt.ylabel("electric-field L2 norm")
        plt.title(f"Landau damping, C={par.collision_strength}, N={par.N}")
        plt.grid(True, which="both", linestyle=":")
        plt.legend()
        fig_path = os.path.join("LandauInhomogeneous1p/Figures", f"{par.output_prefix}_E1_decay.png")
        plt.savefig(fig_path, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {fig_path}")

        plt.figure()
        plt.plot(t, data["kinetic"], marker="o", label="kinetic")
        plt.plot(t, data["field"], marker="s", label="field")
        plt.plot(t, data["total_energy"], marker="^", label="total")
        plt.xlabel("time")
        plt.ylabel("energy")
        plt.title("Energy diagnostics")
        plt.grid(True, linestyle=":")
        plt.legend()
        fig_path = os.path.join("LandauInhomogeneous1p/Figures", f"{par.output_prefix}_energy.png")
        plt.savefig(fig_path, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {fig_path}")


def run(par):
    print("Parameters:")
    print(par)
    print(f"Total particles N = {par.N}")
    print(f"eta = {par.eta:.6g}, eps1 = {par.eps1:.6g}, eps2 = {par.eps2:.6g}")
    print(f"Linear rates: gamma_l = {par.gamma_landau_collisionless:.6g}, gamma_l,c = {par.gamma_landau_collisional_correction:.6g}")

    x, v1, v2, w = initialize_particles_landau_damping(par)
    E1, E2, B3 = initialize_fields(x, v1, v2, w, par)

    x, v1, v2, w, E1, E2, B3, hist = time_cycle(x, v1, v2, w, E1, E2, B3, par)
    save_outputs(x, v1, v2, w, E1, E2, B3, hist, par)
    print("Final ||E1||_2:", hist.E1_l2[-1])
    print("Final total energy:", hist.total_energy[-1])
    return x, v1, v2, w, E1, E2, B3, hist


if __name__ == "__main__":
    par = Params(**BASE_KWARGS)
    run(par)
