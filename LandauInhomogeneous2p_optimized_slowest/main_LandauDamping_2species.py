"""
Two-species, space-inhomogeneous C-PIC code for Landau damping in 1D-2V.

Species 1 is the light/electron species, species 2 the heavy/ion species.
This file merges the inhomogeneous one-species C-PIC structure with the
multispecies Landau collision force.

The defaults are small.  Increase Nx, Nv*_*, Nc*, and random_batches only
after checking that the workflow is correct on the target machine.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from Functions.params import Params
from Functions.initialize import initialize_particles_landau_damping_2species, BatchIdGenerator
from Functions.fields import (
    deposit_charge_current_to_primal_grid,
    initialize_fields,
    advance_fields,
    fields_at_particles,
)
from Functions.collisions_numba import compute_collision_acceleration, compute_regularized_density_at_particles
from Functions.diagnostics import History, total_mass_weighted_momentum, kinetic_energy, conservative_rescale_to_target
from Functions.progress import ProgressBar


BASE_KWARGS = dict(
    # Landau damping datum.  Species 1 perturbed; species 2 initially uniform.
    alpha1=0.1,
    alpha2=0.0,
    k=0.5,

    # Common x grid; separate velocity grids for electrons and ions.
    Nx=64,
    Nv1_v1=32,
    Nv1_v2=32,
    Nc1=2,
    Lv1_v1=4.0,
    Lv1_v2=4.0,
    Nv2_v1=32,
    Nv2_v2=32,
    Nc2=2,
    #Lv2_v1=4.0,
    #Lv2_v2=4.0,

    # Species.  The defaults are not real electron/proton values.
    m1=1.0,
    m2=2.0,
    charge1=-1.0,
    charge2=1.0,
    n1=1.0,
    n2=1.0,
    temp1=1.0,
    temp2=1.0,

    # 1D-2V Coulombian Landau kernel: gam=-2.
    gam=-2.0,
    collision_strength=0.1,  # 0.0 skips collisions entirely in the time loop.
    enable_collisions=True,
    auto_B_from_charges_masses=True,
    # If auto_B_from_charges_masses=False, set B11,B21,B12,B22 manually.
    B11=1.0,
    B21=1.0,
    B12=0.04,
    B22=0.04,

    # Compact splines only: 1=hat, 2=quadratic, 3=cubic.
    spline_order=1,

    # Time.
    T=20.0,
    dt=1.0 / 50.0,

    # Landau damping: electrostatic Ampere mode.  Use field_solver='yee' later
    # for Weibel/full electromagnetic tests.
    field_solver="ampere",
    current_model="charge",
    initial_field="analytic_landau",
    remove_mean_E1=True,

    # Collision acceleration speedups.
    random_batch=True,
    random_batches=32,

    # Conservative correction for the collision part only.
    conservative_rescaling=True,

    # Output and reproducibility.
    random_seed=12345,
    record_interval=0.1,
    output_prefix="landau_damping_2species",
)


def _zero_collision_arrays(v11, v21):
    z1 = np.zeros_like(v11)
    z2 = np.zeros_like(v21)
    return z1, z1.copy(), z2, z2.copy(), None, None


def time_cycle(x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par):
    ntot = par.ntot
    steps_per_record = max(1, round(par.record_interval / par.dt))
    rng_batch = np.random.default_rng(None if par.random_seed is None else par.random_seed + 991)
    batcher1 = BatchIdGenerator(par.N1, par, rng_batch)
    batcher2 = BatchIdGenerator(par.N2, par, rng_batch)

    hist = History()
    if par.collision_active:
        ft1, ft2 = compute_regularized_density_at_particles(x1, v11, v12, w1, x2, v21, v22, w2, par)
    else:
        ft1, ft2 = None, None
    hist.append(0.0, x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par, ft1, ft2)

    bar = ProgressBar(ntot, label=f"N1={par.N1}, N2={par.N2} ")
    start = time.time()

    for step in range(1, ntot + 1):
        # Collision force at f^n.  If collision_strength=0, skip the expensive routine.
        if par.collision_active:
            batch1 = batcher1.next()
            batch2 = batcher2.next()
            a11, a12, a21, a22, ft1, ft2, _, _, _, _ = compute_collision_acceleration(
                x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, par
            )
        else:
            a11, a12, a21, a22, ft1, ft2 = _zero_collision_arrays(v11, v21)

        # Charge/current deposition and field update.
        _, J1, J2 = deposit_charge_current_to_primal_grid(
            x1, v11, v12, w1, x2, v21, v22, w2,
            par.charge1, par.charge2, par.current_factor1, par.current_factor2,
            par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, par.rho_background
        )
        E1, E2, B3 = advance_fields(E1, E2, B3, J1, J2, par)

        Ep11, Ep12, Bp13 = fields_at_particles(E1, E2, B3, x1, par)
        Ep21, Ep22, Bp23 = fields_at_particles(E1, E2, B3, x2, par)

        # Field-only update.  Mass enters through q_s/m_s.
        # v11/v21 are not overwritten until all four field-updated arrays are built,
        # so full copies of the old v1 components are unnecessary.
        vf11 = v11 + par.dt * par.q_over_m1 * (Ep11 + v12 * Bp13)
        vf12 = v12 + par.dt * par.q_over_m1 * (Ep12 - v11 * Bp13)
        vf21 = v21 + par.dt * par.q_over_m2 * (Ep21 + v22 * Bp23)
        vf22 = v22 + par.dt * par.q_over_m2 * (Ep22 - v21 * Bp23)

        if par.conservative_rescaling and par.collision_active:
            target_P = total_mass_weighted_momentum(vf11, vf12, w1, vf21, vf22, w2, par)
            target_K = kinetic_energy(vf11, vf12, w1, vf21, vf22, w2, par)
            vc11 = vf11 + par.dt * a11
            vc12 = vf12 + par.dt * a12
            vc21 = vf21 + par.dt * a21
            vc22 = vf22 + par.dt * a22
            v11, v12, v21, v22 = conservative_rescale_to_target(
                vc11, vc12, vc21, vc22, w1, w2, par, target_P, target_K
            )
        else:
            v11 = vf11 + par.dt * a11
            v12 = vf12 + par.dt * a12
            v21 = vf21 + par.dt * a21
            v22 = vf22 + par.dt * a22

        x1 = (x1 + par.dt * v11) % par.Lx
        x2 = (x2 + par.dt * v21) % par.Lx

        if step % steps_per_record == 0 or step == ntot:
            # ft1/ft2 correspond to f^n for inexpensive diagnostics.  Uncomment
            # the next two lines if exact record-time entropy is required.
            # if par.collision_active:
            #     ft1, ft2 = compute_regularized_density_at_particles(x1, v11, v12, w1, x2, v21, v22, w2, par)
            hist.append(step * par.dt, x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par, ft1, ft2)

        bar.update(step)

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.2f} s")
    return x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist


def save_outputs(x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist, par):
    os.makedirs("LandauInhomogeneous2p/Results", exist_ok=True)
    os.makedirs("LandauInhomogeneous2p/Figures", exist_ok=True)
    data = hist.as_dict()

    if par.save_results:
        path = os.path.join("LandauInhomogeneous2p/Results", f"{par.output_prefix}.npz")
        np.savez(
            path,
            x1=x1, v11=v11, v12=v12, w1=w1,
            x2=x2, v21=v21, v22=v22, w2=w2,
            E1=E1, E2=E2, B3=B3,
            **data,
        )
        print(f"Saved results to {path}")

    if par.make_plots:
        t = data["t"]
        E = np.maximum(data["E1_l2"], 1.0e-300)
        plt.figure()
        plt.semilogy(t, E, marker="o", label="||E1||_2")
        if E.size > 0:
            ref = E[0] * np.exp(par.gamma_landau_collisionless * t)
            plt.semilogy(t, ref, linestyle="--", label="electron collisionless reference")
        plt.xlabel("time")
        plt.ylabel("electric-field L2 norm")
        plt.title(f"Two-species Landau damping, C={par.collision_strength}, N={par.N}")
        plt.grid(True, which="both", linestyle=":")
        plt.legend()
        fig_path = os.path.join("LandauInhomogeneous2p/Figures", f"{par.output_prefix}_E1_decay.png")
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
        fig_path = os.path.join("LandauInhomogeneous2p/Figures", f"{par.output_prefix}_energy.png")
        plt.savefig(fig_path, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {fig_path}")

        plt.figure()
        plt.plot(t, data["T1"], marker="o", label="T species 1")
        plt.plot(t, data["T2"], marker="s", label="T species 2")
        plt.plot(t, data["Ttot"], marker="^", label="T total")
        plt.xlabel("time")
        plt.ylabel("temperature")
        plt.title("Temperature diagnostics")
        plt.grid(True, linestyle=":")
        plt.legend()
        fig_path = os.path.join("LandauInhomogeneous2p/Figures", f"{par.output_prefix}_temperature.png")
        plt.savefig(fig_path, dpi=140, bbox_inches="tight")
        print(f"Saved plot to {fig_path}")


def run(par):
    print("Parameters:")
    print(par)
    print(f"Total particles: N1={par.N1}, N2={par.N2}, N={par.N}")
    print(f"eta={par.eta:.6g}")
    print(f"eps species 1=({par.eps1_v1:.6g},{par.eps1_v2:.6g}), species 2=({par.eps2_v1:.6g},{par.eps2_v2:.6g})")
    print(f"charges=({par.charge1},{par.charge2}), masses=({par.m1},{par.m2}), q/m=({par.q_over_m1},{par.q_over_m2})")
    print(f"B coefficients: B11={par.B11}, B21={par.B21}, B12={par.B12}, B22={par.B22}")
    print(f"collision active: {par.collision_active}")

    x1, v11, v12, w1, x2, v21, v22, w2 = initialize_particles_landau_damping_2species(par)
    E1, E2, B3 = initialize_fields(x1, v11, v12, w1, x2, v21, v22, w2, par)

    out = time_cycle(x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par)
    x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist = out
    save_outputs(x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist, par)
    print("Final ||E1||_2:", hist.E1_l2[-1])
    print("Final total energy:", hist.total_energy[-1])
    return x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist


if __name__ == "__main__":
    par = Params(**BASE_KWARGS)
    run(par)
