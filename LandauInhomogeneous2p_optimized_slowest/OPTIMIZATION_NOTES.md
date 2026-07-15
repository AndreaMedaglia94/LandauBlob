# Implementation optimization notes

This version keeps the same C-PIC/Landau discretization and the same default method parameters.  The changes are implementation-level changes intended to reduce constant factors in the existing cell-list and random-batch code.

## Main changes

1. `Functions/collisions_numba.py`
   - Added Numba-specialized fast paths for the default compact linear/hat spline (`spline_order == 1`).
   - Added specialized Step III kernels for the common 1D-2V Coulomb setting (`gam == -2`) and Maxwell setting (`gam == 0`) so the inner pair loop avoids the generic spline-order branch and avoids `r2 ** half_gam` when it can use `1/r2` or a constant coefficient.
   - Kept the existing generic kernels for `spline_order == 2` and `spline_order == 3`, and for non-specialized `gam` values.
   - Precomputes `w/ftilde` before the second compact phase-space pass to avoid a division in the innermost neighbor loop.
   - Reuses particle cell indices computed during cell-list construction so the hot kernels avoid repeated coordinate-to-cell floor operations.

2. `Functions/fields.py`
   - Added hat-spline deposition and interpolation kernels.  For `spline_order == 1`, only the two nonzero grid points are visited instead of scanning the wider generic support loop.
   - The generic deposition/interpolation functions remain available for higher-order compact splines.

3. `Functions/initialize.py` and `main_LandauDamping_2species.py`
   - Added `BatchIdGenerator`, which reuses the permutation and batch-id arrays between time steps.  It resets to `arange` before shuffling, so it matches the old `rng.permutation(N)` batch sequence while avoiding repeated allocation of large arrays.
   - Removed unnecessary full-array velocity copies in the explicit field update.

## Validation performed

For a deterministic small case, the optimized collision acceleration and regularized-density outputs match the original code up to roundoff, with maximum relative differences about `1e-16` to `3e-16`.

For field deposition/interpolation in a small deterministic case, the optimized outputs match the original code up to roundoff, with interpolation exactly matching in the tested case and deposition/current differences at roundoff level.

## Benchmark on this container

These timings exclude first-call Numba compilation and use `NUMBA_NUM_THREADS=4`.

Configuration for the collision benchmark:

```text
Nx=32
Nv1_v1=Nv1_v2=Nv2_v1=Nv2_v2=16
Nc1=Nc2=2
N1=N2=16384
random_batches=8
spline_order=1
gam=-2
```

Median collision-call time:

```text
original:  0.0732 s
optimized: 0.0398 s
speedup:   about 1.8x
```

Configuration for the field benchmark:

```text
Nx=64
Nv1_v1=Nv1_v2=Nv2_v1=Nv2_v2=16
Nc1=Nc2=2
N1=N2=32768
spline_order=1
```

Median field-operation times:

```text
deposit charge/current: original 0.00160 s, optimized 0.000681 s, speedup about 2.3x
interpolate fields for two species: original 0.000702 s, optimized 0.000194 s, speedup about 3.6x
```

Actual speedups will depend on particle count, `random_batches`, spline order, gamma, CPU cache, and `NUMBA_NUM_THREADS`.
