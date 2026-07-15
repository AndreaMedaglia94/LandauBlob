"""
Two-species homogeneous Landau particle method: Coulomb relaxation example.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from Functions.params import Params
from Functions.initialize import initialize_particles_gaussian
from Functions.diagnostics import compute_moments, conservative_rescale
from Functions.progress import ProgressBar

try:
    from Functions.physics_numba_multispecies import compute_rhs
    RHS_BACKEND = "numba"
except Exception:  # pragma: no cover
    from Functions.physics import compute_rhs
    RHS_BACKEND = "numpy"


def time_cycle(vx1, vy1, wg1, vx2, vy2, wg2, par, record_interval=0.2):
    keys = ["t", "T1", "T2", "T", "U1x", "U1y", "U2x", "U2y", "Ux", "Uy", "energy", "momentum_x", "momentum_y"]
    out = {key: [] for key in keys}

    def record(t):
        mom = compute_moments(vx1, vy1, wg1, vx2, vy2, wg2, par)
        out["t"].append(t)
        out["T1"].append(mom["T1"]); out["T2"].append(mom["T2"]); out["T"].append(mom["T"])
        out["U1x"].append(mom["U1"][0]); out["U1y"].append(mom["U1"][1])
        out["U2x"].append(mom["U2"][0]); out["U2y"].append(mom["U2"][1])
        out["Ux"].append(mom["U"][0]); out["Uy"].append(mom["U"][1])
        out["energy"].append(mom["energy"])
        out["momentum_x"].append(mom["momentum"][0]); out["momentum_y"].append(mom["momentum"][1])

    record(0.0)
    target_T = out["T"][-1]
    steps_per_record = max(1, round(record_interval / par.dt))

    bar = ProgressBar(par.ntot, label=f"Coulomb 2p N={par.N}, {RHS_BACKEND} ")
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
            record(t)
        bar.update(step)

    return {k: np.asarray(v) for k, v in out.items()}


def post_process(par, data):
    os.makedirs("LandauHomogeneous2p/Figures", exist_ok=True)
    os.makedirs("LandauHomogeneous2p/Data", exist_ok=True)

    plt.figure()
    plt.plot(data["t"], data["T1"], "k-o", label="T1")
    plt.plot(data["t"], data["T2"], "r-*", label="T2")
    plt.plot(data["t"], data["T"], "b-^", label="T total")
    plt.xlabel("time t")
    plt.ylabel("temperature")
    plt.legend()
    plt.title(f"Coulomb relaxation, rescaling={par.conservative_rescaling}")
    plt.tight_layout()
    plt.savefig("LandauHomogeneous2p/Figures/Coulomb_temperatures.png", dpi=140)

    plt.figure()
    plt.plot(data["t"], data["U1x"], "k-o", label="U1x")
    plt.plot(data["t"], data["U2x"], "r-*", label="U2x")
    plt.plot(data["t"], data["Ux"], "b-^", label="Ux total")
    plt.xlabel("time t")
    plt.ylabel("velocity")
    plt.legend()
    plt.title(f"Coulomb relaxation, rescaling={par.conservative_rescaling}")
    plt.tight_layout()
    plt.savefig("LandauHomogeneous2p/Figures/Coulomb_velocities_x.png", dpi=140)

    plt.figure()
    plt.plot(data["t"], data["U1y"], "k-o", label="U1y")
    plt.plot(data["t"], data["U2y"], "r-*", label="U2y")
    plt.plot(data["t"], data["Uy"], "b-^", label="Uy total")
    plt.xlabel("time t")
    plt.ylabel("velocity")
    plt.legend()
    plt.title(f"Coulomb relaxation, rescaling={par.conservative_rescaling}")
    plt.tight_layout()
    plt.savefig("LandauHomogeneous2p/Figures/Coulomb_velocities_y.png", dpi=140)

    plt.figure()
    plt.plot(data["t"], data["energy"] - data["energy"][0], "k-o")
    plt.xlabel("time t")
    plt.ylabel("energy - energy(0)")
    plt.title("Total energy error")
    plt.tight_layout()
    plt.savefig("LandauHomogeneous2p/Figures/Coulomb_energy_error.png", dpi=140)

    np.savez("LandauHomogeneous2p/Data/Coulomb_multispecies_output.npz", **data)
    print("Saved LandauHomogeneous2p/Figures/Coulomb_temperatures.png, LandauHomogeneous2p/Figures/Coulomb_energy_error.png, and LandauHomogeneous2p/Data/Coulomb_multispecies_output.npz")


if __name__ == "__main__":
    par = Params(
        gam=-3.0,
        T=50.0,
        dt=0.02,
        Nv=30,
        m1=1.0,
        m2=2.0,
        B11=1.0 / 16.0,
        B21=1.0 / 16.0,
        B12=1.0 / 16.0,
        B22=1.0 / 8.0,
        temp_1_0=1.0 / 8.0,
        temp_2_0=1.0 / 4.0,
        u_1_x_0=-1.0 / 4.0,
        u_1_y_0=0.0,
        u_2_x_0=1.0 / 2.0,
        u_2_y_0=1.0 / 4.0,
        conservative_rescaling=True,
    )

    rng = np.random.default_rng(par.random_seed)
    vx1, vy1, wg1 = initialize_particles_gaussian(par, species=1, rng=rng)
    vx2, vy2, wg2 = initialize_particles_gaussian(par, species=2, rng=rng)
    data = time_cycle(vx1, vy1, wg1, vx2, vy2, wg2, par)
    post_process(par, data)
