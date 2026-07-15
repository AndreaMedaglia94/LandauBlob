# Two-species homogeneous Landau particle code

This folder is a Python translation/extension of the provided MATLAB `BKW_2D_pulito` code, organized with the same structure as the one-species Python code:

```text
main_BKW_multispecies.py        # BKW exact-solution test
main_Coulomb_multispecies.py    # Coulomb relaxation diagnostic test
Functions/params.py             # all physical/numerical parameters
Functions/initialize.py         # BKW and Maxwellian particle initialization
Functions/physics.py            # blocked NumPy RHS
Functions/physics_numba_multispecies.py  # parallel Numba RHS
Functions/reconstruct.py        # blob reconstruction
Functions/diagnostics.py        # moments and conservative rescaling
Functions/kernels.py            # Gaussian and B-spline mollifiers
Functions/progress.py           # progress bar
```

## Conservative rescaling switch

The MATLAB code always applies a post-collision rescaling to restore total thermal energy.  In this Python version it is controlled at the beginning of the run through:

```python
par = Params(conservative_rescaling=True)
```

Set it to `False` to run the plain forward-Euler particle method without that correction:

```python
par = Params(conservative_rescaling=False)
```

The correction is implemented in `Functions/diagnostics.py` as

```python
v <- U + sqrt(T_target / T_current) * (v - U)
```

for both species, where `U` is the current total bulk velocity.  This preserves total momentum and restores the total thermal temperature, hence the total kinetic energy associated with the previous target temperature.

## Running

From inside this folder:

```bash
python main_BKW_multispecies.py
```

or

```bash
python main_Coulomb_multispecies.py
```

The scripts save figures in `Figures/` and arrays in `Data/`.

## Notes

The fast backend uses Numba if it is available.  If the import fails, the main scripts fall back to the blocked NumPy implementation in `Functions/physics.py`.

For the Gaussian mollifier, `Params` uses `epsi_i = 0.64 h_i^1.98`.  When `Lv2` is left as `None`, the default `Lv2 = Lv1 * (m1/m2)^(1/1.98)` enforces `m1*epsi1 = m2*epsi2` for the two species when `shape=0`, matching the condition used in the paper for species-independent equilibrium temperature.
