"""
Regression test: confirms the optimized package (Functions/) produces the
same physics as the original, preserved package (Functions_reference/) --
same method, same random-batch draws, same initial data -- to floating
point round-off.

Run this once after installing numba, before trusting the optimized code
for real work:

    python test_regression.py

It runs several small-but-real configurations (different spline orders,
mass ratios, random-batch counts, and both the gam=-2 fast path and the
general pow()-based path) through both packages and compares every
returned array. Differences should be at the 1e-12..1e-15 level (floating
point summation-order noise from the cell-sort and self+cross fusion,
exactly like switching numpy's reduction order); anything larger indicates
a real problem and should be reported before using the optimized code.

This script was already run against a hand-written pure-Python line-for-
line mirror of the *actual* original algorithm during development (see the
development notes in OPTIMIZATION_NOTES.md); this file re-confirms the
same property using the real, compiled numba code in your environment,
with your numba version, your CPU, and your thread count.
"""
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


def compare(name, a, b, tol=1e-9):
    if a is None and b is None:
        return True
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    maxabs = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = maxabs < tol
    print(f"    [{'OK ' if ok else 'FAIL'}] {name:6s} maxabs_diff={maxabs:.3e}")
    return ok


def run_kernel_case(ref, new, seed, random_batches, spline_order, gam, nx, nv1, nv2, nc1, nc2,
                     m1=1.0, m2=3.0, sort_oversample=1):
    common = dict(
        Nx=nx, Nv1_v1=nv1, Nv1_v2=nv1, Nc1=nc1, Lv1_v1=4.0, Lv1_v2=4.0,
        Nv2_v1=nv2, Nv2_v2=nv2, Nc2=nc2,
        m1=m1, m2=m2, charge1=-1.0, charge2=1.0, n1=1.0, n2=1.0,
        temp1=1.0, temp2=1.0, gam=gam, collision_strength=0.37,
        spline_order=spline_order, random_batch=True, random_batches=random_batches,
        random_seed=seed, dt=1.0 / 50.0, T=1.0,
    )
    par_r = ref["Params"](**common)
    x1, v11, v12, w1, x2, v21, v22, w2 = ref["initialize"].initialize_particles_landau_damping_2species(par_r)
    rng_batch = np.random.default_rng(seed + 991)
    batch1 = ref["initialize"].make_batch_ids(par_r.N1, par_r, rng_batch)
    batch2 = ref["initialize"].make_batch_ids(par_r.N2, par_r, rng_batch)

    out_ref = ref["collisions"].compute_collision_acceleration(
        x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, par_r)
    rho_r, J1_r, J2_r = ref["fields"].deposit_charge_current_to_primal_grid(
        x1, v11, v12, w1, x2, v21, v22, w2,
        par_r.charge1, par_r.charge2, par_r.current_factor1, par_r.current_factor2,
        par_r.Lx, par_r.eta, par_r.Nx, par_r.spline_order, par_r.spline_radius, par_r.rho_background)

    par_n = new["Params"](sort_oversample=sort_oversample, **common)
    out_new = new["collisions"].compute_collision_acceleration(
        x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, par_n)
    rho_n, J1_n, J2_n = new["fields"].deposit_charge_current_to_primal_grid(
        x1, v11, v12, w1, x2, v21, v22, w2,
        par_n.charge1, par_n.charge2, par_n.current_factor1, par_n.current_factor2,
        par_n.Lx, par_n.eta, par_n.Nx, par_n.spline_order, par_n.spline_radius, par_n.rho_background,
        par_n.sort_oversample)

    print(f"  seed={seed} R={random_batches} order={spline_order} gam={gam} Nx={nx} "
          f"N1={par_r.N1} N2={par_r.N2} sort_oversample={sort_oversample}")
    names = ["a11", "a12", "a21", "a22", "ft1", "ft2", "F11", "F12", "F21", "F22"]
    ok = True
    for nm, a, b in zip(names, out_ref, out_new):
        ok &= compare(nm, a, b)
    ok &= compare("rho", rho_r, rho_n)
    ok &= compare("J1", J1_r, J1_n)
    ok &= compare("J2", J2_r, J2_n)
    return ok


def run_full_loop_case(ref, new, nsteps=8):
    from Functions_reference.diagnostics import (
        total_mass_weighted_momentum as tmwm_ref, kinetic_energy as ke_ref,
        conservative_rescale_to_target as crt_ref,
    )
    from Functions.diagnostics import (
        total_mass_weighted_momentum as tmwm_new, kinetic_energy as ke_new,
        conservative_rescale_to_target as crt_new,
    )

    def loop(mod, tmwm, ke, crt, sort_oversample=None):
        kwargs = dict(
            Nx=6, Nv1_v1=4, Nv1_v2=4, Nc1=2, Lv1_v1=4.0, Lv1_v2=4.0,
            Nv2_v1=4, Nv2_v2=4, Nc2=2,
            m1=1.0, m2=4.0, charge1=-1.0, charge2=1.0, n1=1.0, n2=1.0,
            temp1=1.0, temp2=1.0, gam=-2.0, collision_strength=0.3,
            spline_order=1, random_batch=True, random_batches=6,
            random_seed=2024, dt=1.0 / 40.0, T=1.0,
            conservative_rescaling=True, field_solver="ampere",
            initial_field="analytic_landau",
        )
        if sort_oversample is not None:
            kwargs["sort_oversample"] = sort_oversample
        par = mod["Params"](**kwargs)
        x1, v11, v12, w1, x2, v21, v22, w2 = mod["initialize"].initialize_particles_landau_damping_2species(par)
        E1, E2, B3 = mod["fields"].initialize_fields(x1, v11, v12, w1, x2, v21, v22, w2, par)
        rng_batch = np.random.default_rng(par.random_seed + 991)

        for _ in range(nsteps):
            batch1 = mod["initialize"].make_batch_ids(par.N1, par, rng_batch)
            batch2 = mod["initialize"].make_batch_ids(par.N2, par, rng_batch)
            a11, a12, a21, a22, ft1, ft2, *_ = mod["collisions"].compute_collision_acceleration(
                x1, v11, v12, w1, x2, v21, v22, w2, batch1, batch2, par)
            if sort_oversample is not None:
                _, J1, J2 = mod["fields"].deposit_charge_current_to_primal_grid(
                    x1, v11, v12, w1, x2, v21, v22, w2,
                    par.charge1, par.charge2, par.current_factor1, par.current_factor2,
                    par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, par.rho_background,
                    par.sort_oversample)
            else:
                _, J1, J2 = mod["fields"].deposit_charge_current_to_primal_grid(
                    x1, v11, v12, w1, x2, v21, v22, w2,
                    par.charge1, par.charge2, par.current_factor1, par.current_factor2,
                    par.Lx, par.eta, par.Nx, par.spline_order, par.spline_radius, par.rho_background)
            E1, E2, B3 = mod["fields"].advance_fields(E1, E2, B3, J1, J2, par)
            Ep11, Ep12, Bp13 = mod["fields"].fields_at_particles(E1, E2, B3, x1, par)
            Ep21, Ep22, Bp23 = mod["fields"].fields_at_particles(E1, E2, B3, x2, par)

            vf11 = v11 + par.dt * par.q_over_m1 * (Ep11 + v12 * Bp13)
            vf12 = v12 + par.dt * par.q_over_m1 * (Ep12 - v11 * Bp13)
            vf21 = v21 + par.dt * par.q_over_m2 * (Ep21 + v22 * Bp23)
            vf22 = v22 + par.dt * par.q_over_m2 * (Ep22 - v21 * Bp23)

            target_P = tmwm(vf11, vf12, w1, vf21, vf22, w2, par)
            target_K = ke(vf11, vf12, w1, vf21, vf22, w2, par)
            vc11 = vf11 + par.dt * a11
            vc12 = vf12 + par.dt * a12
            vc21 = vf21 + par.dt * a21
            vc22 = vf22 + par.dt * a22
            v11, v12, v21, v22 = crt(vc11, vc12, vc21, vc22, w1, w2, par, target_P, target_K)

            x1 = (x1 + par.dt * v11) % par.Lx
            x2 = (x2 + par.dt * v21) % par.Lx

        return x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3

    out_r = loop(ref, tmwm_ref, ke_ref, crt_ref, sort_oversample=None)
    out_n = loop(new, tmwm_new, ke_new, crt_new, sort_oversample=2)

    names = ["x1", "v11", "v12", "w1", "x2", "v21", "v22", "w2", "E1", "E2", "B3"]
    print(f"  {nsteps}-step full trajectory (collisions + fields + push):")
    ok = True
    for nm, a, b in zip(names, out_r, out_n):
        ok &= compare(nm, a, b)
    return ok


if __name__ == "__main__":
    try:
        import numba  # noqa: F401
    except ImportError:
        print("numba is not installed in this environment -- install requirements.txt first:")
        print("    pip install -r requirements.txt")
        sys.exit(1)

    print(f"numba threads available: {numba.get_num_threads()}")
    ref, new = _import_pair()

    print("\n=== Kernel-level regression (Steps I-III + deposition) ===")
    t0 = time.time()
    trials = [
        dict(seed=0, random_batches=4, spline_order=1, gam=-2.0, nx=5, nv1=4, nv2=4, nc1=2, nc2=2),
        dict(seed=1, random_batches=4, spline_order=2, gam=-2.0, nx=5, nv1=4, nv2=4, nc1=2, nc2=2,
             sort_oversample=3),
        dict(seed=2, random_batches=8, spline_order=3, gam=-2.0, nx=5, nv1=4, nv2=4, nc1=2, nc2=2),
        dict(seed=3, random_batches=4, spline_order=1, gam=-1.5, nx=5, nv1=4, nv2=4, nc1=2, nc2=2),
        dict(seed=4, random_batches=1, spline_order=1, gam=-2.0, nx=5, nv1=4, nv2=4, nc1=2, nc2=2),
        dict(seed=5, random_batches=9, spline_order=1, gam=-2.0, nx=3, nv1=3, nv2=3, nc1=3, nc2=3,
             m1=1.0, m2=10.0),
    ]
    all_ok = True
    for t in trials:
        all_ok &= run_kernel_case(ref, new, **t)
    print(f"(kernel checks took {time.time()-t0:.1f}s -- includes first-call numba JIT compilation)")

    print("\n=== Full-loop regression (physical trajectory) ===")
    all_ok &= run_full_loop_case(ref, new)

    print("\n" + "=" * 60)
    if all_ok:
        print("REGRESSION PASSED: optimized package matches the original.")
    else:
        print("REGRESSION FAILED: see FAIL lines above. Do not trust results")
        print("from Functions/ until this is resolved -- please report this.")
        sys.exit(1)
