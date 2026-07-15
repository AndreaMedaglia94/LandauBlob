# LandauInhomogeneous2p

Two-species, space-inhomogeneous 1D-2V C-PIC code for a Landau-damping test.

Species 1 is the light/electron species and species 2 is the heavy/ion species.
The spatial grid is common to both species.  Velocity-domain half-widths and
velocity resolutions are separate for the two species, so the ion particles can
live on a different velocity grid.

Run:

```bash
python main_LandauDamping_2species.py
```

The default parameters are intentionally small.  The main file contains a
`BASE_KWARGS` block for changing the run.

Important options:

```python
collision_strength = 0.0
```

skips all collision computations in the main time loop.

```python
random_batch = True
random_batches = 8
```

uses random batching in Step III of the collision operator.  Steps I-II always
use compact phase-space cell lists.

```python
conservative_rescaling = True
```

applies a global post-collision correction that restores the kinetic energy and
mass-weighted momentum of the field-only update.

```python
field_solver = "ampere"
```

runs the electrostatic Landau-damping mode.  `field_solver="yee"` is included for
later electromagnetic tests.

The particle weights are number-density weights.  With the default
`current_model="charge"`, Ampere uses

```text
J = sum_s q_s int v f_s dv.
```

The masses enter the Lorentz acceleration through `q_s/m_s`.  For an alternative
normalisation where the current should be weighted by `q_s/m_s`, set

```python
current_model = "charge_over_mass"
```

By default

```python
auto_B_from_charges_masses = True
```

sets the multispecies collision coefficients with the physical scaling
`B_ji ~ q_i^2 q_j^2 / m_i`.  Thus for cross-species collisions
`m1*B21 = m2*B12`, which is the symmetry needed by the multispecies Landau
pair update.  Set it to `False` only when you want to prescribe `B11`, `B21`,
`B12`, and `B22` manually.
