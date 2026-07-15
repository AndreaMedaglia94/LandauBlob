"""
Particle method for the homogeneous Landau equation (2D BKW test case).
par is the only thing you should need to edit between test cases (see params.py).

Structure: initialize params -> initialize particles -> time cycle -> post-processing.
"""
import numpy as np
import matplotlib.pyplot as plt

from Functions.params import Params
from Functions.initialize import initialize_particles_det_BKW, initialize_particles_stratified_BKW, bkw_2d
### change here to switch to parallelized numba implementation of compute_rhs
#from Functions.physics import compute_rhs
from Functions.physics_numba_general import compute_rhs
###
from Functions.reconstruct import reconstruct
from Functions.progress import ProgressBar


def time_cycle(vx, vy, wg, par, record_interval=0.1):
    ntot = par.ntot
    steps_per_record = max(1, round(record_interval / par.dt))
 
    times, L1_error, L2_error = [], [], []
 
    def record(t):
        f_exact = bkw_2d(t, par.temp, par.c, par.beta, par.grid_x, par.grid_y)
        f_num = reconstruct(vx, vy, wg, par.grid_x, par.grid_y, par)
        l2 = np.sqrt(np.sum(par.h**2 * (f_num - f_exact) ** 2) / np.sum(par.h**2 * f_exact**2))
        l1 = np.sum(par.h**2 * np.abs(f_num - f_exact)) / np.sum(par.h**2 * np.abs(f_exact))
        times.append(t)
        L1_error.append(l1)
        L2_error.append(l2)
 
    record(0.0)
 
    bar = ProgressBar(ntot, label=f"N={par.N} ")
    for step in range(1, ntot + 1):
        t = par.dt * step
        dvx, dvy = compute_rhs(vx, vy, wg, par)
        vx = vx + par.dt * dvx
        vy = vy + par.dt * dvy
 
        if step % steps_per_record == 0 or step == ntot:
            record(t)
        bar.update(step)
 
    return vx, vy, wg, np.array(times), np.array(L1_error), np.array(L2_error)


def post_process(par, times, L1_error, L2_error):
    plt.figure()
    plt.semilogy(times, L2_error, "k-o", label="L2 error")
    plt.semilogy(times, L1_error, "r-*", label="L1 error")
    plt.xlabel("time t")
    plt.legend()
    plt.title(f"N={par.N}, shape={par.shape}")
    plt.savefig("LandauHomogeneous1p/Figures/Lerror_plot.png", dpi=120)
    print("Final L2 error:", L2_error[-1])
    print("Final L1 error:", L1_error[-1])


if __name__ == "__main__":
    # 1. initialize params
    par = Params()

    # 2. initialize particles
    #vx, vy, wg = initialize_particles_det_BKW(par)
    vx, vy, wg = initialize_particles_stratified_BKW(par)

    # 3. time cycle
    vx, vy, wg, times,L1_error, L2_error = time_cycle(vx, vy, wg, par, record_interval=0.1)

    # 4. post-processing
    post_process(par, times, L1_error, L2_error)