"""
Parameter object for the one-species, space-inhomogeneous 1D-2V
Vlasov-Maxwell-Landau C-PIC code.

This file plays the same role as params.py in the homogeneous code:
edit Params(...) in the main script, or change defaults here.

The defaults are deliberately smaller than the paper run, so that a first
Python test finishes in a reasonable time.  The paper Landau-damping
parameters are recorded in main_LandauDamping.py.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
import math
import numpy as np


@dataclass
class Params:
    # ------------------------------------------------------------------
    # Problem: Landau damping in 1D-2V.
    # f0(x,v1,v2) = (1 + alpha*cos(k*x))/(2*pi) * exp(-(v1^2+v2^2)/2)
    # The spatial domain is [0,Lx), with Lx=2*pi/k unless overridden.
    # ------------------------------------------------------------------
    alpha: float = 0.1
    k: float = 0.5
    Lx: Optional[float] = None
    Lv1: float = 4.0
    Lv2: float = 4.0

    # ------------------------------------------------------------------
    # Mesh and particle resolution.
    # Total particles N = Nx * Nv1 * Nv2 * Nc.
    # eta = Lx/Nx, eps1 = Lv1/Nv1, eps2 = Lv2/Nv2, following section 2.2.4.
    # ------------------------------------------------------------------
    Nx: int = 32
    Nv1: int = 12
    Nv2: int = 12
    Nc: int = 2

    # ------------------------------------------------------------------
    # Landau collision kernel A(z) = C * |z|^gam * (|z|^2 I - z otimes z).
    # In 1D-2V Coulombian tests, gam = -2.
    # Set collision_strength=0.0 for collisionless PIC.
    # ------------------------------------------------------------------
    gam: float = -2.0
    collision_strength: float = 0.1
    enable_collisions: bool = True

    # ------------------------------------------------------------------
    # Compact B-spline regularisation order.
    # No Gaussian option is included here: compact support is essential for
    # cell-list acceleration in the inhomogeneous algorithm.
    # order=1 is the hat function used in section 2.2.1.
    # ------------------------------------------------------------------
    spline_order: int = 1

    # ------------------------------------------------------------------
    # Time stepping.
    # ------------------------------------------------------------------
    T: float = 2.0
    dt: float = 0.02

    # ------------------------------------------------------------------
    # Field solver.
    # "ampere" updates only E1_t = -J1, with E2=B3=0.  This is the
    # Vlasov-Ampere-Landau setting used for the Landau damping test.
    # "yee" uses the 1D-2V Yee staggered Maxwell update for E1,E2,B3.
    # ------------------------------------------------------------------
    field_solver: Literal["ampere", "yee"] = "ampere"
    rho_ion: float = 1.0
    initial_field: Literal["analytic_landau", "poisson_particles"] = "analytic_landau"

    # ------------------------------------------------------------------
    # Random batch optimisation for the third collision step U[f].
    # Steps I-II are always computed with compact phase-space cell lists.
    # Step III is local in x but nonlocal in v; batching reduces this cost.
    # ------------------------------------------------------------------
    random_batch: bool = True
    random_batches: int = 8

    # ------------------------------------------------------------------
    # Initial particle sampling.
    # Positions are stratified in x. Velocities are sampled from a normal
    # distribution, optionally rejection-truncated to the fictitious velocity box.
    # ------------------------------------------------------------------
    random_seed: Optional[int] = 12345
    truncate_velocity_samples: bool = True
    enforce_zero_momentum: bool = True

    # ------------------------------------------------------------------
    # Diagnostics and output.
    # ------------------------------------------------------------------
    record_interval: float = 0.1
    save_results: bool = True
    make_plots: bool = True
    output_prefix: str = "landau_damping"

    # ------------------------------------------------------------------
    # Derived quantities.
    # ------------------------------------------------------------------
    N: int = field(init=False, repr=False)
    eta: float = field(init=False, repr=False)
    eps1: float = field(init=False, repr=False)
    eps2: float = field(init=False, repr=False)
    x_grid: np.ndarray = field(init=False, repr=False)
    x_faces: np.ndarray = field(init=False, repr=False)
    spline_radius: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.k <= 0.0:
            raise ValueError("k must be positive")
        if self.Lx is None:
            self.Lx = 2.0 * math.pi / self.k
        if self.Lx <= 0.0 or self.Lv1 <= 0.0 or self.Lv2 <= 0.0:
            raise ValueError("Lx, Lv1 and Lv2 must be positive")
        if self.Nx <= 0 or self.Nv1 <= 0 or self.Nv2 <= 0 or self.Nc <= 0:
            raise ValueError("Nx, Nv1, Nv2 and Nc must be positive")
        if self.dt <= 0.0 or self.T < 0.0:
            raise ValueError("dt must be positive and T must be non-negative")
        if self.spline_order not in (1, 2, 3):
            raise ValueError("spline_order must be 1, 2, or 3; Gaussian regularisation is not used here")
        if self.field_solver not in ("ampere", "yee"):
            raise ValueError("field_solver must be 'ampere' or 'yee'")
        if self.initial_field not in ("analytic_landau", "poisson_particles"):
            raise ValueError("initial_field must be 'analytic_landau' or 'poisson_particles'")
        if self.random_batches <= 0:
            raise ValueError("random_batches must be positive")

        self.N = self.Nx * self.Nv1 * self.Nv2 * self.Nc
        self.eta = self.Lx / self.Nx
        self.eps1 = self.Lv1 / self.Nv1
        self.eps2 = self.Lv2 / self.Nv2
        self.spline_radius = {1: 1.0, 2: 1.5, 3: 2.0}[self.spline_order]

        if self.random_batch and self.random_batches > 1:
            if self.random_batches >= self.N:
                raise ValueError("random_batches must be smaller than N")
            if self.N % self.random_batches != 0:
                raise ValueError(
                    "For the exact random-batch scaling, N must be divisible by random_batches. "
                    f"Here N={self.N}, random_batches={self.random_batches}."
                )

        self.x_grid = (np.arange(self.Nx, dtype=float) + 0.5) * self.eta
        self.x_faces = np.arange(self.Nx, dtype=float) * self.eta

    @property
    def ntot(self) -> int:
        return math.ceil(self.T / self.dt)

    @property
    def particles_per_x_cell(self) -> int:
        return self.Nv1 * self.Nv2 * self.Nc

    @property
    def gamma_landau_collisionless(self) -> float:
        return -(1.0 / self.k**3) * math.sqrt(math.pi / 8.0) * math.exp(-1.0 / (2.0 * self.k**2) - 1.5)

    @property
    def gamma_landau_collisional_correction(self) -> float:
        return -math.sqrt(2.0 / (9.0 * math.pi))
