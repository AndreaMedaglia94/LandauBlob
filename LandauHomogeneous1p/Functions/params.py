"""
All physical & numerical parameters for a single run live here, along
with everything derived from them (mesh, evaluation grid, mollifier
bandwidth). Constructing a Params object fully sets up a run -- nothing
mesh-related should need to be computed in main.py anymore.

This is the ONLY file you should need to touch to set up a new test
case (analogous to editing SetParam.m).
"""
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np


@dataclass
class Params:
    # --- collision kernel A(z) = beta * |z|^gam * (|z|^2 I - z ⊗ z) ---
    gam: float = 0.0      # kernel exponent gamma (Maxwell molecules: 0)
    beta: float = 1 / 16  # kernel prefactor

    # --- velocity domain / mesh ---
    Lv: float = 4.0        # domain is [-Lv, Lv]^2
    Nv: int = 40        # cells per dimension
    Npc: int = 1          # particles per cell (N = Nv^2 * Npc)

    # --- time stepping ---
    T: float = 5.0
    dt: float = 0.01

    # --- BKW exact-solution constants (2D Maxwell molecules) ---
    c: float = 0.5     # "C_BKW"
    temp: float = 1.0  # "T_tot"

    # --- mollifier shape: 0=Gaussian, 1/2/3 = B-spline order ---
    shape: int = 3

    # --- performance ---
    # Row-block size for the chunked pairwise sums in physics.py /
    # reconstruct.py. Peak memory scales as O(block_size * N), not
    # O(N^2), so this is the knob that keeps large N from exploding.
    # Bigger = faster (fewer, larger BLAS calls) but more memory;
    # tune it to whatever fits comfortably in RAM.
    block_size: int = 1000

    # --- derived quantities -----------------------------------------
    # Computed once in __post_init__ from the fields above -- do not
    # set these directly when constructing a Params. Kept as real
    # (not @property) attributes so they're computed once and reused,
    # not recomputed on every access. repr=False keeps print(par)
    # readable instead of dumping full arrays.
    dV: float = field(init=False, repr=False, default=None)
    Vc: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    h: float = field(init=False, repr=False, default=None)
    epsi: float = field(init=False, repr=False, default=None)
    grid_x: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    grid_y: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        V = np.linspace(-self.Lv, self.Lv, self.Nv + 1)
        self.dV = V[1] - V[0]
        self.Vc = V[:-1] + self.dV / 2
        self.h = 2 * self.Lv / self.Nv

        self.epsi = 0.64 * self.h**1.98 if self.shape == 0 else self.h

        # single consistent 'ij'-indexed grid, used both as the initial
        # particle positions and as the evaluation grid for error/blob
        # reconstruction (see note previously in initialize.py)
        VXc, VYc = np.meshgrid(self.Vc, self.Vc, indexing="ij")
        self.grid_x = VXc.ravel()
        self.grid_y = VYc.ravel()

    @property
    def N(self) -> int:
        return self.Nv ** 2 * self.Npc

    @property
    def ntot(self) -> int:
        return math.ceil(self.T / self.dt)