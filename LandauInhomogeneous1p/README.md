# LandauInhomogeneous1p

One-species, space-inhomogeneous 1D-2V C-PIC code for the Vlasov-Maxwell-Landau equation, starting with the Landau damping test of section 3.1.2 of Bailo, Carrillo and Hu (2024).

The code follows the structure of the homogeneous Python code:

```text
LandauInhomogeneous1p/
  main_LandauDamping.py
  Functions/
    params.py
    initialize.py
    fields.py
    collisions_numba.py
    splines_numba.py
    diagnostics.py
    progress.py
  Figures/
  Results/
```

## Main features

- 1D in space and 2D in velocity.
- One particle species.
- Compact tensor-product B-spline regularisation only: order 1, 2, or 3.
- No Gaussian regularisation for the collision operator.
- Phase-space cell list for the compactly supported steps I-II of the collision operator.
- Spatial cell list for step III of the collision operator.
- Optional random batch acceleration for step III.
- Vlasov-Ampere field solver for Landau damping.
- Optional full 1D-2V Yee staggered Maxwell update for E1, E2, B3.

## Run

From inside the folder:

```bash
python main_LandauDamping.py
```

The default run is intentionally smaller than the paper run so that it can be tested quickly in Python.
The paper-scale Landau damping parameters are given as comments in `main_LandauDamping.py`:

```python
Nx = 128
Nv1 = Nv2 = 32
Nc = 8
T = 10
dt = 1/50
random_batches = 32
Lv1 = Lv2 = 4
alpha = 0.1
k = 0.5
gam = -2
collision_strength in {0, 0.01, 0.1, 1}
```

That configuration has `N = 1,048,576` particles. It should be treated as a production-scale run, not a first test.

## Important parameters

In `Functions/params.py`:

```python
spline_order = 1       # 1=hat, 2=quadratic, 3=cubic
random_batch = True
random_batches = 8
field_solver = "ampere"  # Landau damping
field_solver = "yee"     # full 1D-2V staggered Maxwell update
```

The collision strength is `collision_strength`. Set it to zero for the collisionless PIC reference.

## Notes on cost

The inhomogeneous collision operator is much more expensive than the homogeneous one. Steps I-II are local in both `x` and `v` because of the compact B-splines. Step III is local in `x` but nonlocal in `v`; random batching reduces that cost. Increase resolution gradually.
