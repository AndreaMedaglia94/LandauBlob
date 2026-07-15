"""
Convergence-in-Nv study for the 2D BKW test case, reproducing the
methodology of Fig. 1 (right panel) in Carrillo, Hu, Wang & Wu (2020):
for a fixed final time T, run the particle method at a series of mesh
resolutions Nv, compute the relative L1/L2/Linf error of the blob
reconstruction against the exact BKW solution at that final time, and
fit a log-log slope in h = 2*Lv/Nv to check for the paper's reported
~2nd-order convergence.

Structure mirrors mainBKW.py: initialize params -> run one simulation
per Nv -> collect final-time errors -> post-process (table + log-log
plot + fitted slopes).

WHY THIS DOESN'T REUSE mainBKW.py's time_cycle(): time_cycle() also
reconstructs the blob solution and computes L1/L2 error at EVERY step
(for the diagnostic error-vs-time plot in mainBKW.py). A convergence
study only needs the error at the FINAL time, and reconstruct() costs
about as much as one compute_rhs() call -- doing it every step would
roughly double the cost of every run in this study, on top of already
running one full simulation per Nv. run_single() below does the same
forward-Euler stepping but only reconstructs once, at the end.

Edit NV_LIST below to whatever resolutions you want to test.
"""
import numpy as np
import matplotlib.pyplot as plt

from Functions.params import Params
from Functions.initialize import initialize_particles_det_BKW, initialize_particles_stratified_BKW, bkw_2d
### change here to switch to the parallelized numba implementation of compute_rhs
# from physics import compute_rhs
from Functions.physics_numba_general import compute_rhs
###
from Functions.reconstruct import reconstruct
from Functions.progress import ProgressBar


# ----------------------------------------------------------------------
# EDIT HERE: resolutions to test, and the settings shared by every run.
# Only Nv changes between runs -- everything else comes from one
# Params(), so runs differ ONLY in resolution, matching the paper.
NV_LIST = [20, 40, 60, 80]
BASE_KWARGS = dict(T=5.0, dt=0.01, gam=0.0, shape=3, Lv=4.0, Npc=4)
# ----------------------------------------------------------------------


def final_time_errors(vx, vy, wg, par, t_final):
    """Relative L1, L2, Linf error of the blob reconstruction against the
    exact BKW solution at t_final. Same norm definitions as mainBKW.py's
    time_cycle() (L1/L2) and Section 4 of the paper (Linf: max over the
    grid, no h-weighting, since it's a sup-norm not an integral norm)."""
    f_exact = bkw_2d(t_final, par.temp, par.c, par.beta, par.grid_x, par.grid_y)
    f_num = reconstruct(vx, vy, wg, par.grid_x, par.grid_y, par)
    diff = f_num - f_exact

    L2 = np.sqrt(np.sum(par.h**2 * diff**2) / np.sum(par.h**2 * f_exact**2))
    L1 = np.sum(par.h**2 * np.abs(diff)) / np.sum(par.h**2 * np.abs(f_exact))
    Linf = np.max(np.abs(diff)) / np.max(np.abs(f_exact))
    return L1, L2, Linf


def run_single(Nv):
    """Run one full simulation at the given Nv, return (h, L1, L2, Linf)
    at the final time T."""
    par = Params(Nv=Nv, **BASE_KWARGS)
    #vx, vy, wg = initialize_particles_det_BKW(par)
    vx, vy, wg = initialize_particles_stratified_BKW(par)

    ntot = par.ntot
    bar = ProgressBar(ntot, label=f"Nv={Nv:4d} (N={par.N:6d}) ")
    for step in range(1, ntot + 1):
        dvx, dvy = compute_rhs(vx, vy, wg, par)
        vx = vx + par.dt * dvx
        vy = vy + par.dt * dvy
        bar.update(step)

    L1, L2, Linf = final_time_errors(vx, vy, wg, par, par.T)
    return par.h, L1, L2, Linf


def fit_slope(h_vals, err_vals):
    """Least-squares slope of log(err) vs log(h) -- matches the paper's
    'using the least square fitting, we can find the approximate slope
    of the errors'."""
    slope, _ = np.polyfit(np.log(h_vals), np.log(err_vals), 1)
    return slope


def post_process(hs, L1s, L2s, Linfs):
    slopes = None
    if len(hs) >= 2:
        slopes = (fit_slope(hs, L1s), fit_slope(hs, L2s), fit_slope(hs, Linfs))
        print()
        print("Fitted convergence order (slope of log-log least-squares fit):")
        print(f"  L1:   {slopes[0]:.2f}")
        print(f"  L2:   {slopes[1]:.2f}")
        print(f"  Linf: {slopes[2]:.2f}")
        print("(paper reports ~2nd order for this test case)")
    else:
        print()
        print("Need at least 2 Nv values to fit a convergence slope.")

    plt.figure()
    label_L1 = f"L1 (slope {slopes[0]:.2f})" if slopes else "L1"
    label_L2 = f"L2 (slope {slopes[1]:.2f})" if slopes else "L2"
    label_Linf = f"Linf (slope {slopes[2]:.2f})" if slopes else "Linf"
    plt.loglog(hs, L1s, "r-*", label=label_L1)
    plt.loglog(hs, L2s, "k-o", label=label_L2)
    plt.loglog(hs, Linfs, "b-^", label=label_Linf)
    plt.xlabel("h")
    plt.ylabel("relative error")
    plt.title(f"Convergence in Nv (T={BASE_KWARGS['T']}, dt={BASE_KWARGS['dt']}, shape={BASE_KWARGS['shape']})")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.savefig("LandauHomogeneous1p/Figures/convergence_plot.png", dpi=120)
    print()
    print("Saved plot to LandauHomogeneous1p/Figures/convergence_plot.png")


if __name__ == "__main__":
    # 1. initialize params (implicitly, one per Nv, inside run_single)
    print(f"Convergence study: Nv = {NV_LIST}")
    print(f"Fixed settings: {BASE_KWARGS}")
    print()

    # 2+3. initialize particles + time cycle, per Nv
    results = [run_single(Nv) for Nv in NV_LIST]

    hs, L1s, L2s, Linfs = (np.array(x) for x in zip(*results))
    print()
    print(f'{"Nv":>5} {"h":>10} {"L1":>12} {"L2":>12} {"Linf":>12}')
    for Nv, h, L1, L2, Linf in zip(NV_LIST, hs, L1s, L2s, Linfs):
        print(f"{Nv:5d} {h:10.5f} {L1:12.4e} {L2:12.4e} {Linf:12.4e}")

    # 4. post-processing
    post_process(hs, L1s, L2s, Linfs)