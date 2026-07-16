"""
Two-species, space-inhomogeneous C-PIC code for Landau damping in 1D-2V.
"""

import os
import time
import numpy as np
from scipy.io import savemat

from Functions.params import Params, params_to_matlab_dict
from Functions.initialize import initialize_particles_landau_damping_2species, BatchIdGenerator
from Functions.fields import (
    deposit_charge_current_to_primal_grid,
    initialize_fields,
    advance_fields,
    fields_at_particles,
)
from Functions.collisions_numba import compute_collision_acceleration, compute_regularized_density_at_particles
from Functions.diagnostics import History, EnergyLog, total_mass_weighted_momentum, kinetic_energy, conservative_rescale_to_target
from Functions.reconstruct import DistributionLog, compute_all_marginals
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

    # Species.
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
    # If auto_B_from_charges_masses=False, set B11,B21,B12,B22 manually.
    auto_B_from_charges_masses=False,
    B11=1.0,
    B21=1.0,
    B12=0.04,
    B22=0.04,

    # Compact splines only: 1=hat, 2=quadratic, 3=cubic.
    spline_order=3,

    # Time.
    T=20.0,
    dt=1.0 / 50.0,

    # Landau damping: electrostatic Ampere mode.
    field_solver="ampere",
    current_model="charge",
    initial_field="analytic_landau",
    remove_mean_E1=True,

    # Collision acceleration speedups.
    random_batch=True,
    random_batches=8,

    # Conservative correction for the collision part only.
    conservative_rescaling=True,

    # Output and reproducibility.
    random_seed=12345,
    record_interval=0.1,        # cadence for the History diagnostics
    dist_save_interval=0.5,     # cadence for the .mat-saved f(x,v) marginal snapshots
    save_results=True,
    save_distributions=True,
    output_prefix="landau_damping_2species",

    # Reconstruction resolution for the saved/plotted marginals, decoupled
    # from the physics velocity resolution (Nv1_v1, ... above, which also
    # set the particle count). Leave commented out to reuse Nv1_v1/Nv1_v2/
    # Nv2_v1/Nv2_v2 as before. Uncomment and raise for a finer diagnostic
    # grid -- this does NOT change the number of particles.
    # dist_Nv1_v1=64,
    # dist_Nv1_v2=64,
    # dist_Nv2_v1=64,
    # dist_Nv2_v2=64,
)

# ----------------------------------------------------------------------
# Figures: independent of Params/BASE_KWARGS -- purely a driver-script
# concern, so a cluster batch run never has to think about matplotlib.
# MAKE_PLOTS=False skips all plotting AND the extra per-step/per-snapshot
# bookkeeping it needs (no overhead when it's off).
# ----------------------------------------------------------------------
MAKE_PLOTS = False
PLOT_MARGINAL_TIMES = [0.0, 5.0, 10.0, 15.0, 20.0]  # physical times for the
# f(x,v) marginal figures; distinct from BASE_KWARGS['dist_save_interval'],
# which only controls the cadence of the .mat-saved snapshots.


def _zero_collision_arrays(v11, v21):
    z1 = np.zeros_like(v11)
    z2 = np.zeros_like(v21)
    return z1, z1.copy(), z2, z2.copy(), None, None


class _SpeciesEnergyLog:
    """Per-species kinetic energy at every step. Plotting-only -- kept
    local to this driver rather than in Functions/diagnostics.py, since
    the saved .mat 'energy' struct only carries the combined kinetic
    energy (Functions.diagnostics.EnergyLog)."""
    def __init__(self):
        self.t = []
        self.KE1 = []
        self.KE2 = []

    def append(self, t, v11, v12, w1, v21, v22, w2, par):
        ke1 = 0.5 * par.m1 * np.sum(w1 * (v11 * v11 + v12 * v12))
        ke2 = 0.5 * par.m2 * np.sum(w2 * (v21 * v21 + v22 * v22))
        self.t.append(float(t))
        self.KE1.append(float(ke1))
        self.KE2.append(float(ke2))


def time_cycle(x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par,
               profile=True, make_plots=False, plot_marginal_times=None):
    ntot = par.ntot
    steps_per_record = max(1, round(par.record_interval / par.dt))
    steps_per_dist = max(1, round(par.dist_save_interval / par.dt))
    rng_batch = np.random.default_rng(None if par.random_seed is None else par.random_seed + 991)
    batcher1 = BatchIdGenerator(par.N1, par, rng_batch)
    batcher2 = BatchIdGenerator(par.N2, par, rng_batch)

    hist = History()
    elog = EnergyLog()
    dist = DistributionLog() if par.save_distributions else None
    spelog = _SpeciesEnergyLog() if make_plots else None

    # Map requested marginal-plot times to the nearest achievable step.
    plot_steps = set()
    plot_marginals = {"t": [], "f1_vx": [], "f1_vy": [], "f2_vx": [], "f2_vy": [],
                       "x_grid": None, "v1_v1_grid": None, "v1_v2_grid": None,
                       "v2_v1_grid": None, "v2_v2_grid": None}
    if make_plots and plot_marginal_times:
        for t_req in plot_marginal_times:
            plot_steps.add(max(0, min(ntot, int(round(t_req / par.dt)))))

    def _maybe_capture_plot_marginals(step, x1, v11, v12, w1, x2, v21, v22, w2):
        if step not in plot_steps:
            return
        m = compute_all_marginals(x1, v11, v12, w1, x2, v21, v22, w2, par)
        if plot_marginals["x_grid"] is None:
            for g in ("x_grid", "v1_v1_grid", "v1_v2_grid", "v2_v1_grid", "v2_v2_grid"):
                plot_marginals[g] = m[g]
        plot_marginals["t"].append(step * par.dt)
        for key in ("f1_vx", "f1_vy", "f2_vx", "f2_vy"):
            plot_marginals[key].append(m[key])

    if par.collision_active:
        ft1, ft2 = compute_regularized_density_at_particles(x1, v11, v12, w1, x2, v21, v22, w2, par)
    else:
        ft1, ft2 = None, None
    hist.append(0.0, x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par, ft1, ft2)
    elog.append(0.0, v11, v12, w1, v21, v22, w2, E1, E2, B3, par)
    if spelog is not None:
        spelog.append(0.0, v11, v12, w1, v21, v22, w2, par)
    if dist is not None:
        dist.append(0.0, x1, v11, v12, w1, x2, v21, v22, w2, par)
    _maybe_capture_plot_marginals(0, x1, v11, v12, w1, x2, v21, v22, w2)

    bar = ProgressBar(ntot, label=f"N1={par.N1}, N2={par.N2} ")
    start = time.time()
    t_collision = 0.0
    t_fields = 0.0
    t_push = 0.0

    for step in range(1, ntot + 1):
        # Collision force at f^n.  If collision_strength=0, skip the expensive routine.
        t0 = time.time()
        if par.collision_active:
            batch1 = batcher1.next()
            batch2 = batcher2.next()
            a11, a12, a21, a22, ft1, ft2, _, _, _, _ = compute_collision_acceleration(
                x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, par
            )
        else:
            a11, a12, a21, a22, ft1, ft2 = _zero_collision_arrays(v11, v21)
        t1 = time.time()

        # Charge/current deposition and field update.
        _, J1, J2 = deposit_charge_current_to_primal_grid(
            x1, v11, v12, w1, x2, v21, v22, w2,
            par.charge1, par.charge2, par.current_factor1, par.current_factor2,
            par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, par.rho_background,
            par.sort_oversample
        )
        E1, E2, B3 = advance_fields(E1, E2, B3, J1, J2, par)

        Ep11, Ep12, Bp13 = fields_at_particles(E1, E2, B3, x1, par)
        Ep21, Ep22, Bp23 = fields_at_particles(E1, E2, B3, x2, par)
        t2 = time.time()

        # Field-only update.  Mass enters through q_s/m_s.  v11/v21 are read
        # here before being rebound below, so no defensive copy is needed.
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
        t3 = time.time()

        t_collision += t1 - t0
        t_fields += t2 - t1
        t_push += t3 - t2

        # Energies every step: cheap O(Nx)+O(N) sums, negligible next to
        # the collision/deposition work above.
        elog.append(step * par.dt, v11, v12, w1, v21, v22, w2, E1, E2, B3, par)
        if spelog is not None:
            spelog.append(step * par.dt, v11, v12, w1, v21, v22, w2, par)

        if step % steps_per_record == 0 or step == ntot:
            # ft1/ft2 correspond to f^n for inexpensive diagnostics.  Uncomment
            # the next two lines if exact record-time entropy is required.
            # if par.collision_active:
            #     ft1, ft2 = compute_regularized_density_at_particles(x1, v11, v12, w1, x2, v21, v22, w2, par)
            hist.append(step * par.dt, x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par, ft1, ft2)

        if dist is not None and (step % steps_per_dist == 0 or step == ntot):
            dist.append(step * par.dt, x1, v11, v12, w1, x2, v21, v22, w2, par)

        _maybe_capture_plot_marginals(step, x1, v11, v12, w1, x2, v21, v22, w2)

        bar.update(step)

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.2f} s")
    if profile and elapsed > 0.0:
        print(f"  collisions (Steps I-III): {t_collision:6.2f} s  ({100*t_collision/elapsed:4.1f}%)")
        print(f"  deposit + field solve:    {t_fields:6.2f} s  ({100*t_fields/elapsed:4.1f}%)")
        print(f"  push + rescale + wrap:    {t_push:6.2f} s  ({100*t_push/elapsed:4.1f}%)")
    return x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist, elog, spelog, dist, plot_marginals


def save_outputs(hist, elog, dist, par):
    """
    Write a single .mat file with macroscopic/kinetic diagnostics and the
    run parameters. Never writes particle arrays or figures.
    """
    if not par.save_results:
        return

    os.makedirs("LandauInhomogeneous2p/Results", exist_ok=True)
    path = os.path.join("LandauInhomogeneous2p/Results", f"{par.output_prefix}.mat")

    mdict = {
        "energy": elog.as_dict(),            # every time step
        "diagnostics": hist.as_dict(),        # every par.record_interval
        "par": params_to_matlab_dict(par),
    }
    if par.save_distributions and dist is not None:
        mdict["distributions"] = dist.as_dict()  # every par.dist_save_interval

    savemat(path, mdict, do_compression=True)
    print(f"Saved results to {path}")


def save_figures(elog, spelog, plot_marginals, par):
    """
    Optional PNG figures (only called when make_plots=True). matplotlib is
    imported lazily here, so a make_plots=False run never needs it
    installed at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("LandauInhomogeneous2p/Figures", exist_ok=True)
    e = elog.as_dict()
    t = e["t"]
    KE1 = np.asarray(spelog.KE1)
    KE2 = np.asarray(spelog.KE2)

    # 1) All energies together: electric, kinetic per species, total.
    plt.figure()
    plt.plot(t, e["E_field"], label="electric")
    plt.plot(t, KE1, label="kinetic species 1")
    plt.plot(t, KE2, label="kinetic species 2")
    plt.plot(t, e["E_total"], label="total", linewidth=2)
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.title(f"All energies, C={par.collision_strength}, N={par.N}")
    plt.grid(True, linestyle=":")
    plt.legend()
    fig_path = os.path.join("LandauInhomogeneous2p/Figures", f"{par.output_prefix}_all_energies.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {fig_path}")

    # 2) Electric-field energy alone, with the collisionless reference.
    # Energy ~ (E1_l2)^2, and E1_l2 decays like exp(gamma*t), so the
    # energy reference uses exp(2*gamma*t).
    plt.figure()
    Efield = np.maximum(e["E_field"], 1.0e-300)
    plt.semilogy(t, Efield, marker="o", markersize=2, label="electric-field energy")
    if Efield.size > 0 and Efield[0] > 0.0:
        ref = Efield[0] * np.exp(2.0 * par.gamma_landau_collisionless * t)
        plt.semilogy(t, ref, linestyle="--", label="collisionless reference (2*gamma)")
    plt.xlabel("time")
    plt.ylabel("electric-field energy")
    plt.title("Electric-field energy")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    fig_path = os.path.join("LandauInhomogeneous2p/Figures", f"{par.output_prefix}_electric_energy.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {fig_path}")

    # 3) Kinetic energies of the two species.
    plt.figure()
    plt.plot(t, KE1, label="species 1 (KE1)")
    plt.plot(t, KE2, label="species 2 (KE2)")
    plt.xlabel("time")
    plt.ylabel("kinetic energy")
    plt.title("Per-species kinetic energy")
    plt.grid(True, linestyle=":")
    plt.legend()
    fig_path = os.path.join("LandauInhomogeneous2p/Figures", f"{par.output_prefix}_kinetic_species.png")
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {fig_path}")

    # 4) Reconstructed marginals at the requested times: one 2x2 figure
    # per snapshot (f1_vx, f1_vy, f2_vx, f2_vy).
    if plot_marginals["t"]:
        x = plot_marginals["x_grid"]
        panels = (
            ("f1_vx", plot_marginals["v1_v1_grid"], "species 1, v1"),
            ("f1_vy", plot_marginals["v1_v2_grid"], "species 1, v2"),
            ("f2_vx", plot_marginals["v2_v1_grid"], "species 2, v1"),
            ("f2_vy", plot_marginals["v2_v2_grid"], "species 2, v2"),
        )
        for i, t_snap in enumerate(plot_marginals["t"]):
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for ax, (key, vgrid, title) in zip(axes.ravel(), panels):
                f = plot_marginals[key][i]
                pcm = ax.pcolormesh(x, vgrid, f.T, shading="auto")
                ax.set_xlabel("x")
                ax.set_ylabel("v")
                ax.set_title(title)
                fig.colorbar(pcm, ax=ax)
            fig.suptitle(f"Velocity marginals at t={t_snap:.4g}")
            fig_path = os.path.join(
                "LandauInhomogeneous2p/Figures",
                f"{par.output_prefix}_marginals_t{t_snap:.4g}.png",
            )
            fig.savefig(fig_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot to {fig_path}")


def run(par, make_plots=MAKE_PLOTS, plot_marginal_times=PLOT_MARGINAL_TIMES):
    print("Parameters:")
    print(par)
    print(f"Total particles: N1={par.N1}, N2={par.N2}, N={par.N}")
    print(f"eta={par.eta:.6g}")
    print(f"eps species 1=({par.eps1_v1:.6g},{par.eps1_v2:.6g}), species 2=({par.eps2_v1:.6g},{par.eps2_v2:.6g})")
    print(f"charges=({par.charge1},{par.charge2}), masses=({par.m1},{par.m2}), q/m=({par.q_over_m1},{par.q_over_m2})")
    print(f"B coefficients: B11={par.B11}, B21={par.B21}, B12={par.B12}, B22={par.B22}")
    print(f"collision active: {par.collision_active}")
    print(f"make_plots: {make_plots}")

    x1, v11, v12, w1, x2, v21, v22, w2 = initialize_particles_landau_damping_2species(par)
    E1, E2, B3 = initialize_fields(x1, v11, v12, w1, x2, v21, v22, w2, par)

    out = time_cycle(
        x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par,
        make_plots=make_plots, plot_marginal_times=plot_marginal_times,
    )
    x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist, elog, spelog, dist, plot_marginals = out

    save_outputs(hist, elog, dist, par)
    if make_plots:
        save_figures(elog, spelog, plot_marginals, par)

    print("Final ||E1||_2 (record_interval grid):", hist.E1_l2[-1])
    print("Final total energy (record_interval grid):", hist.total_energy[-1])
    print("Final total energy (every step):", elog.E_total[-1])
    return x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, hist, elog, dist


if __name__ == "__main__":
    par = Params(**BASE_KWARGS)
    run(par)
