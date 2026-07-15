"""
Two-species homogeneous Landau particle method: BKW test case.

Structure mirrors the one-species Python code:
    initialize params -> initialize particles -> time cycle -> post-processing.

The boolean par.conservative_rescaling controls the MATLAB-style correction
applied after each explicit Euler collision step.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from Functions.params import Params
from Functions.initialize import initialize_particles_stratified_BKW, initialize_particles_det_BKW, bkw_2d
from Functions.reconstruct import reconstruct
from Functions.diagnostics import compute_moments, conservative_rescale
from Functions.progress import ProgressBar

try:
    from Functions.physics_numba_multispecies import compute_rhs
    RHS_BACKEND = "numba"
except Exception:  # pragma: no cover - only used when numba is unavailable
    from Functions.physics import compute_rhs
    RHS_BACKEND = "numpy"


def relative_errors(vx, vy, wg, grid_x, grid_y, h, epsi, mass, density, par, t):
    f_exact = bkw_2d(t, par.temp, par.c, par.beta, grid_x, grid_y, mass, density)
    f_num = reconstruct(vx, vy, wg, grid_x, grid_y, par.shape, epsi, par.block_size)
    diff = f_num - f_exact
    L2 = np.sqrt(np.sum(h**2 * diff**2) / np.sum(h**2 * f_exact**2))
    L1 = np.sum(h**2 * np.abs(diff)) / np.sum(h**2 * np.abs(f_exact))
    Linf = np.max(np.abs(diff)) / np.max(np.abs(f_exact))
    return L1, L2, Linf


def record_state(t, vx1, vy1, wg1, vx2, vy2, wg2, par, out):
    e1 = relative_errors(vx1, vy1, wg1, par.grid_x1, par.grid_y1, par.h1, par.epsi1, par.m1, par.n1, par, t)
    e2 = relative_errors(vx2, vy2, wg2, par.grid_x2, par.grid_y2, par.h2, par.epsi2, par.m2, par.n2, par, t)
    mom = compute_moments(vx1, vy1, wg1, vx2, vy2, wg2, par)

    out["t"].append(t)
    out["L1_1"].append(e1[0]); out["L2_1"].append(e1[1]); out["Linf_1"].append(e1[2])
    out["L1_2"].append(e2[0]); out["L2_2"].append(e2[1]); out["Linf_2"].append(e2[2])
    out["T1"].append(mom["T1"]); out["T2"].append(mom["T2"]); out["T"].append(mom["T"])
    out["U1x"].append(mom["U1"][0]); out["U1y"].append(mom["U1"][1])
    out["U2x"].append(mom["U2"][0]); out["U2y"].append(mom["U2"][1])
    out["Ux"].append(mom["U"][0]); out["Uy"].append(mom["U"][1])
    out["energy"].append(mom["energy"])
    out["momentum_x"].append(mom["momentum"][0]); out["momentum_y"].append(mom["momentum"][1])


def time_cycle(vx1, vy1, wg1, vx2, vy2, wg2, par, record_interval=0.1):
    out = {key: [] for key in [
        "t", "L1_1", "L2_1", "Linf_1", "L1_2", "L2_2", "Linf_2",
        "T1", "T2", "T", "U1x", "U1y", "U2x", "U2y", "Ux", "Uy",
        "energy", "momentum_x", "momentum_y",
    ]}

    record_state(0.0, vx1, vy1, wg1, vx2, vy2, wg2, par, out)
    target_T = out["T"][-1]
    steps_per_record = max(1, round(record_interval / par.dt))

    bar = ProgressBar(par.ntot, label=f"2p N={par.N}, {RHS_BACKEND} ")
    for step in range(1, par.ntot + 1):
        t = min(step * par.dt, par.T)
        dvx1, dvy1, dvx2, dvy2 = compute_rhs(vx1, vy1, wg1, vx2, vy2, wg2, par)
        vx1 = vx1 + par.dt * dvx1
        vy1 = vy1 + par.dt * dvy1
        vx2 = vx2 + par.dt * dvx2
        vy2 = vy2 + par.dt * dvy2

        if par.conservative_rescaling:
            vx1, vy1, vx2, vy2 = conservative_rescale(vx1, vy1, wg1, vx2, vy2, wg2, par, target_T)

        if step % steps_per_record == 0 or step == par.ntot:
            record_state(t, vx1, vy1, wg1, vx2, vy2, wg2, par, out)
        bar.update(step)

    return vx1, vy1, wg1, vx2, vy2, wg2, {k: np.asarray(v) for k, v in out.items()}


def post_process(par, data):
    os.makedirs("LandauHomogeneous2p/Figures", exist_ok=True)
    os.makedirs("LandauHomogeneous2p/Data", exist_ok=True)

    plt.figure()
    plt.semilogy(data["t"], data["L2_1"], "k-o", label="species 1 L2")
    plt.semilogy(data["t"], data["L2_2"], "r-*", label="species 2 L2")
    plt.xlabel("time t")
    plt.ylabel("relative L2 error")
    plt.legend()
    plt.title(f"Two-species BKW, rescaling={par.conservative_rescaling}")
    plt.tight_layout()
    plt.savefig("LandauHomogeneous2p/Figures/BKW_L2_errors.png", dpi=140)

    plt.figure()
    plt.plot(data["t"], data["T1"], "k-o", label="T1")
    plt.plot(data["t"], data["T2"], "r-*", label="T2")
    plt.plot(data["t"], data["T"], "b-^", label="T total")
    plt.xlabel("time t")
    plt.ylabel("temperature")
    plt.legend()
    plt.title("Temperatures")
    plt.tight_layout()
    plt.savefig("LandauHomogeneous2p/Figures/BKW_temperatures.png", dpi=140)

    np.savez("LandauHomogeneous2p/Data/BKW_multispecies_output.npz", **data)

    print("Final relative errors:")
    print(f"  species 1: L1={data['L1_1'][-1]:.6e}, L2={data['L2_1'][-1]:.6e}, Linf={data['Linf_1'][-1]:.6e}")
    print(f"  species 2: L1={data['L1_2'][-1]:.6e}, L2={data['L2_2'][-1]:.6e}, Linf={data['Linf_2'][-1]:.6e}")
    print(f"Saved LandauHomogeneous2p/Figures/BKW_L2_errors.png, LandauHomogeneous2p/Figures/BKW_temperatures.png, and LandauHomogeneous2p/Data/BKW_multispecies_output.npz")


if __name__ == "__main__":
    # Edit parameters here.  conservative_rescaling=True reproduces the MATLAB
    # post-collision correction; False gives the plain explicit Euler update.
    par = Params(conservative_rescaling=True)

    rng = np.random.default_rng(par.random_seed)
    # Use deterministic center particles by replacing the next two lines with:
    # vx1, vy1, wg1 = initialize_particles_det_BKW(par, species=1)
    # vx2, vy2, wg2 = initialize_particles_det_BKW(par, species=2)
    vx1, vy1, wg1 = initialize_particles_stratified_BKW(par, species=1, rng=rng)
    vx2, vy2, wg2 = initialize_particles_stratified_BKW(par, species=2, rng=rng)

    vx1, vy1, wg1, vx2, vy2, wg2, data = time_cycle(vx1, vy1, wg1, vx2, vy2, wg2, par)
    post_process(par, data)
