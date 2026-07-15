"""
Benchmark: wall-clock comparison of Functions_reference/ (original,
linked-list cell lists) vs Functions/ (optimized, cell-sorted + fused Step
III + pow-avoidance + parallel deposition), on identical workloads.

Run after `python test_regression.py` has passed, so you know the two are
computing the same thing:

    python benchmark.py                 # default sweep
    python benchmark.py --quick         # smaller/faster sweep
    python benchmark.py --threads 1 4 8 # also compare thread counts

Notes on reading the numbers:
  * The first call to any @njit function pays a one-time compilation cost
    (worse for parallel=True functions). This script always does a warm-up
    call before timing, so reported times are steady-state.
  * numba.set_num_threads() controls the *available* thread pool; the OS
    scheduler and BLAS/OMP env vars can still interfere. For a clean
    reading, run with an explicit thread count, e.g.
    NUMBA_NUM_THREADS=8 python benchmark.py
  * The collision step (Steps I-III) is the dominant cost for any
    non-trivial problem size -- see the per-phase timing breakdown that
    main_LandauDamping_2species.py also prints.
"""
import argparse
import sys
import time
import numpy as np

sys.path.insert(0, ".")


def _import_pair():
    import Functions_reference.params as params_ref
    import Functions_reference.initialize as initialize_ref
    import Functions_reference.collisions_numba as collisions_ref
    import Functions_reference.fields as fields_ref

    import Functions.params as params_new
    import Functions.initialize as initialize_new
    import Functions.collisions_numba as collisions_new
    import Functions.fields as fields_new

    ref = dict(Params=params_ref.Params, initialize=initialize_ref, collisions=collisions_ref, fields=fields_ref)
    new = dict(Params=params_new.Params, initialize=initialize_new, collisions=collisions_new, fields=fields_new)
    return ref, new


def make_case(mod, seed, random_batches, spline_order, nx, nv1, nv2, nc1, nc2,
              m1=1.0, m2=25.0, sort_oversample=None):
    kwargs = dict(
        Nx=nx, Nv1_v1=nv1, Nv1_v2=nv1, Nc1=nc1, Lv1_v1=4.0, Lv1_v2=4.0,
        Nv2_v1=nv2, Nv2_v2=nv2, Nc2=nc2,
        m1=m1, m2=m2, charge1=-1.0, charge2=1.0, n1=1.0, n2=1.0,
        temp1=1.0, temp2=1.0, gam=-2.0, collision_strength=0.1,
        spline_order=spline_order, random_batch=True, random_batches=random_batches,
        random_seed=seed, dt=1.0 / 50.0, T=1.0,
    )
    if sort_oversample is not None:
        kwargs["sort_oversample"] = sort_oversample
    par = mod["Params"](**kwargs)
    x1, v11, v12, w1, x2, v21, v22, w2 = mod["initialize"].initialize_particles_landau_damping_2species(par)
    rng_batch = np.random.default_rng(seed + 991)
    batch1 = mod["initialize"].make_batch_ids(par.N1, par, rng_batch)
    batch2 = mod["initialize"].make_batch_ids(par.N2, par, rng_batch)
    return par, x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2


def time_collision_step(mod, par, x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, repeats):
    # warm-up / JIT compile
    mod["collisions"].compute_collision_acceleration(x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, par)
    t0 = time.perf_counter()
    for _ in range(repeats):
        mod["collisions"].compute_collision_acceleration(
            x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, par)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def time_deposit_step(mod, par, x1, v11, v12, w1, x2, v21, v22, w2, repeats, sort_oversample=None):
    args = (x1, v11, v12, w1, x2, v21, v22, w2,
            par.charge1, par.charge2, par.current_factor1, par.current_factor2,
            par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, par.rho_background)
    extra = () if sort_oversample is None else (par.sort_oversample,)
    mod["fields"].deposit_charge_current_to_primal_grid(*args, *extra)
    t0 = time.perf_counter()
    for _ in range(repeats):
        mod["fields"].deposit_charge_current_to_primal_grid(*args, *extra)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def run_sweep(configs, repeats):
    ref, new = _import_pair()
    print(f"{'config':<28s}{'N1':>9s}{'N2':>9s}{'orig(ms)':>12s}{'opt(ms)':>12s}{'speedup':>10s}")
    for cfg in configs:
        label = cfg.pop("label")
        par_r, x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2 = make_case(ref, **cfg)
        t_ref = time_collision_step(ref, par_r, x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, repeats)

        par_n, x1n, v11n, v12n, w1n, x2n, v21n, v22n, w2n, batch1n, batch2n = make_case(
            new, sort_oversample=cfg.get("sort_oversample", 1), **cfg)
        t_new = time_collision_step(new, par_n, x1n, v11n, v12n, w1n, x2n, v21n, v22n, w2n,
                                     batch1n, batch2n, repeats)

        speedup = t_ref / t_new if t_new > 0 else float("nan")
        print(f"{label:<28s}{par_r.N1:>9d}{par_r.N2:>9d}{1000*t_ref:>12.2f}{1000*t_new:>12.2f}{speedup:>9.2f}x")
        cfg["label"] = label


def run_deposit_sweep(configs, repeats):
    ref, new = _import_pair()
    print(f"\nDeposition-only timing:")
    print(f"{'config':<28s}{'N1':>9s}{'N2':>9s}{'orig(ms)':>12s}{'opt(ms)':>12s}{'speedup':>10s}")
    for cfg in configs:
        label = cfg["label"]
        cfg_no_label = {k: v for k, v in cfg.items() if k != "label"}
        par_r, x1, v11, v12, w1, x2, v21, v22, w2, _, _ = make_case(ref, **cfg_no_label)
        t_ref = time_deposit_step(ref, par_r, x1, v11, v12, w1, x2, v21, v22, w2, repeats)

        par_n, x1n, v11n, v12n, w1n, x2n, v21n, v22n, w2n, _, _ = make_case(
            new, sort_oversample=cfg.get("sort_oversample", 1), **cfg_no_label)
        t_new = time_deposit_step(new, par_n, x1n, v11n, v12n, w1n, x2n, v21n, v22n, w2n, repeats,
                                   sort_oversample=True)

        speedup = t_ref / t_new if t_new > 0 else float("nan")
        print(f"{label:<28s}{par_r.N1:>9d}{par_r.N2:>9d}{1000*t_ref:>12.3f}{1000*t_new:>12.3f}{speedup:>9.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="smaller/faster sweep")
    p.add_argument("--threads", type=int, nargs="*", default=None,
                    help="also repeat the sweep at each of these numba thread counts")
    args = p.parse_args()

    try:
        import numba
    except ImportError:
        print("numba is not installed -- install requirements.txt first: pip install -r requirements.txt")
        sys.exit(1)

    if args.quick:
        configs = [
            dict(label="small (Nx=16,Nv=8)", seed=0, random_batches=8, spline_order=1,
                 nx=16, nv1=8, nv2=8, nc1=2, nc2=2),
            dict(label="medium (Nx=32,Nv=12)", seed=0, random_batches=16, spline_order=1,
                 nx=32, nv1=12, nv2=12, nc1=2, nc2=2),
        ]
        repeats = 3
    else:
        configs = [
            dict(label="small (Nx=16,Nv=8)", seed=0, random_batches=8, spline_order=1,
                 nx=16, nv1=8, nv2=8, nc1=2, nc2=2),
            dict(label="medium (Nx=32,Nv=16)", seed=0, random_batches=16, spline_order=1,
                 nx=32, nv1=16, nv2=16, nc1=2, nc2=2),
            dict(label="large (Nx=64,Nv=24)", seed=0, random_batches=32, spline_order=1,
                 nx=64, nv1=24, nv2=24, nc1=2, nc2=2),
            dict(label="spline_order=3", seed=0, random_batches=16, spline_order=3,
                 nx=32, nv1=16, nv2=16, nc1=2, nc2=2),
            dict(label="R=1 (no batching)", seed=0, random_batches=1, spline_order=1,
                 nx=32, nv1=12, nv2=12, nc1=2, nc2=2),
            dict(label="large mass ratio", seed=0, random_batches=16, spline_order=1,
                 nx=32, nv1=16, nv2=16, nc1=2, nc2=2, m2=100.0),
        ]
        repeats = 5

    if args.threads:
        for nt in args.threads:
            numba.set_num_threads(nt)
            print(f"\n### numba threads = {numba.get_num_threads()} ###")
            run_sweep([dict(c) for c in configs], repeats)
    else:
        print(f"numba threads available: {numba.get_num_threads()} "
              f"(set NUMBA_NUM_THREADS or pass --threads to change)")
        run_sweep([dict(c) for c in configs], repeats)

    run_deposit_sweep([dict(c) for c in configs], repeats=max(3, repeats))
