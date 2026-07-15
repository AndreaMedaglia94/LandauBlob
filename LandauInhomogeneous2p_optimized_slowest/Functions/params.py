"""
Parameter object for the two-species, space-inhomogeneous 1D-2V
Vlasov-Maxwell-Landau C-PIC code.

Species 1 is intended to be the light species/electrons and species 2 the
heavy species/ions.  The code keeps separate velocity resolutions and
velocity-domain half-widths for the two species, but a common spatial grid.

The defaults are deliberately modest.  They are intended for testing the
workflow, not for a paper-scale run.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
import math
import numpy as np


@dataclass
class Params:
    # ------------------------------------------------------------------
    # Landau damping perturbation in x.
    # Species 1 is perturbed by default; species 2 is initially uniform.
    # Spatial domain is [0,Lx), with Lx=2*pi/k unless overridden.
    # ------------------------------------------------------------------
    alpha1: float = 0.1
    alpha2: float = 0.0
    k: float = 0.5
    Lx: Optional[float] = None

    # ------------------------------------------------------------------
    # Common spatial grid and species-specific velocity grids.
    # Total particles:
    #   N1 = Nx * Nv1_v1 * Nv1_v2 * Nc1
    #   N2 = Nx * Nv2_v1 * Nv2_v2 * Nc2
    # eta = Lx/Nx, eps_s_d = Lv_s_d/Nv_s_d.
    # ------------------------------------------------------------------
    Nx: int = 32

    Nv1_v1: int = 12
    Nv1_v2: int = 12
    Nc1: int = 2
    Lv1_v1: float = 4.0
    Lv1_v2: float = 4.0

    Nv2_v1: int = 10
    Nv2_v2: int = 10
    Nc2: int = 2
    Lv2_v1: float = 1.0
    Lv2_v2: float = 1.0

    # Optional helper for heavy species: enforce m1*eps1 = m2*eps2 in each
    # velocity direction, assuming eps = Lv/Nv.  If True, Lv2_* is overwritten.
    auto_Lv2_equal_m_epsi: bool = True

    # ------------------------------------------------------------------
    # Species parameters.  The particle weights represent number density.
    # charge1=-1, charge2=+1 with n1=n2 gives global neutrality.
    # ------------------------------------------------------------------
    m1: float = 1.0
    m2: float = 25.0
    charge1: float = -1.0
    charge2: float = 1.0
    n1: float = 1.0
    n2: float = 1.0

    # Maxwellian initial velocity data.  Thermal speed is sqrt(temp_i/m_i).
    temp1: float = 1.0
    temp2: float = 1.0
    u1_v1_0: float = 0.0
    u1_v2_0: float = 0.0
    u2_v1_0: float = 0.0
    u2_v2_0: float = 0.0

    # ------------------------------------------------------------------
    # Landau collision kernel.  B21 acts on species 1 by source species 2;
    # B12 acts on species 2 by source species 1.
    # In 1D-2V Coulombian tests, gam=-2.
    # collision_strength=0.0 skips the collision routine in the main driver.
    # ------------------------------------------------------------------
    gam: float = -2.0
    collision_strength: float = 0.1
    enable_collisions: bool = True
    auto_B_from_charges_masses: bool = True
    B11: float = 1.0
    B21: float = 1.0
    B12: float = 0.04
    B22: float = 0.04

    # Compact B-spline regularisation order.  No Gaussian option is used here.
    spline_order: int = 1

    # ------------------------------------------------------------------
    # Time stepping and optional conservative correction of the collision part.
    # If conservative_rescaling=True, the post-collision velocities are shifted
    # and scaled so that collisions do not change total kinetic energy or total
    # mass-weighted momentum.  The target state is the field-only update.
    # ------------------------------------------------------------------
    T: float = 2.0
    dt: float = 0.02
    conservative_rescaling: bool = False

    # ------------------------------------------------------------------
    # Field solver.
    # ampere: E1_t = -J1, E2=B3=0.  This is the Landau damping mode.
    # yee:    1D-2V staggered update for E1,E2,B3.
    # current_model='charge' uses J=sum q_s int v f_s dv.
    # current_model='charge_over_mass' uses q_s/m_s instead, for alternative
    # normalisations where weights represent mass-like density.
    # ------------------------------------------------------------------
    field_solver: Literal["ampere", "yee"] = "ampere"
    current_model: Literal["charge", "charge_over_mass"] = "charge"
    rho_background: float = 0.0
    initial_field: Literal["analytic_landau", "poisson_particles"] = "analytic_landau"
    remove_mean_E1: bool = True

    # ------------------------------------------------------------------
    # Random batch optimisation for Step III of the collision operator.
    # Steps I-II are always computed with compact phase-space cell lists.
    # Same-species batching uses the paper scaling R(N-1)/(N-R).  Cross-species
    # batching uses the unbiased scale R.
    # ------------------------------------------------------------------
    random_batch: bool = True
    random_batches: int = 8

    # ------------------------------------------------------------------
    # Initial particle sampling.
    # ------------------------------------------------------------------
    random_seed: Optional[int] = 12345
    truncate_velocity_samples: bool = True
    enforce_zero_species_mean_velocity: bool = True
    enforce_zero_total_current: bool = True

    # ------------------------------------------------------------------
    # Diagnostics and output.
    # ------------------------------------------------------------------
    record_interval: float = 0.1
    save_results: bool = True
    make_plots: bool = False
    output_prefix: str = "landau_damping_2species"

    # ------------------------------------------------------------------
    # Derived quantities.
    # ------------------------------------------------------------------
    N1: int = field(init=False, repr=False)
    N2: int = field(init=False, repr=False)
    N: int = field(init=False, repr=False)
    eta: float = field(init=False, repr=False)
    eps1_v1: float = field(init=False, repr=False)
    eps1_v2: float = field(init=False, repr=False)
    eps2_v1: float = field(init=False, repr=False)
    eps2_v2: float = field(init=False, repr=False)
    x_grid: np.ndarray = field(init=False, repr=False)
    x_faces: np.ndarray = field(init=False, repr=False)
    spline_radius: float = field(init=False, repr=False)
    masses: np.ndarray = field(init=False, repr=False)
    charges: np.ndarray = field(init=False, repr=False)
    densities: np.ndarray = field(init=False, repr=False)
    B: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.k <= 0.0:
            raise ValueError("k must be positive")
        if self.Lx is None:
            self.Lx = 2.0 * math.pi / self.k
        if self.Lx <= 0.0:
            raise ValueError("Lx must be positive")
        if self.Nx <= 0:
            raise ValueError("Nx must be positive")
        if min(self.Nv1_v1, self.Nv1_v2, self.Nv2_v1, self.Nv2_v2, self.Nc1, self.Nc2) <= 0:
            raise ValueError("velocity resolutions and particles per cell must be positive")
        if self.m1 <= 0.0 or self.m2 <= 0.0:
            raise ValueError("species masses must be positive")
        if self.n1 <= 0.0 or self.n2 <= 0.0:
            raise ValueError("species number densities must be positive")
        if self.temp1 < 0.0 or self.temp2 < 0.0:
            raise ValueError("species temperatures must be non-negative")
        if self.dt <= 0.0 or self.T < 0.0:
            raise ValueError("dt must be positive and T must be non-negative")
        if self.spline_order not in (1, 2, 3):
            raise ValueError("spline_order must be 1, 2, or 3")
        if self.field_solver not in ("ampere", "yee"):
            raise ValueError("field_solver must be 'ampere' or 'yee'")
        if self.current_model not in ("charge", "charge_over_mass"):
            raise ValueError("current_model must be 'charge' or 'charge_over_mass'")
        if self.initial_field not in ("analytic_landau", "poisson_particles"):
            raise ValueError("initial_field must be 'analytic_landau' or 'poisson_particles'")
        if self.random_batches <= 0:
            raise ValueError("random_batches must be positive")

        if self.auto_Lv2_equal_m_epsi:
            # eps2 = (m1/m2)*eps1 and eps = Lv/Nv.
            self.Lv2_v1 = self.Lv1_v1 * (self.m1 / self.m2) * (self.Nv2_v1 / self.Nv1_v1)
            self.Lv2_v2 = self.Lv1_v2 * (self.m1 / self.m2) * (self.Nv2_v2 / self.Nv1_v2)

        if min(self.Lv1_v1, self.Lv1_v2, self.Lv2_v1, self.Lv2_v2) <= 0.0:
            raise ValueError("all velocity-domain half-widths must be positive")

        self.N1 = self.Nx * self.Nv1_v1 * self.Nv1_v2 * self.Nc1
        self.N2 = self.Nx * self.Nv2_v1 * self.Nv2_v2 * self.Nc2
        self.N = self.N1 + self.N2
        self.eta = self.Lx / self.Nx
        self.eps1_v1 = self.Lv1_v1 / self.Nv1_v1
        self.eps1_v2 = self.Lv1_v2 / self.Nv1_v2
        self.eps2_v1 = self.Lv2_v1 / self.Nv2_v1
        self.eps2_v2 = self.Lv2_v2 / self.Nv2_v2
        self.spline_radius = {1: 1.0, 2: 1.5, 3: 2.0}[self.spline_order]

        if self.random_batch and self.random_batches > 1:
            R = self.random_batches
            if R >= self.N1 or R >= self.N2:
                raise ValueError("random_batches must be smaller than both N1 and N2")
            if self.N1 % R != 0 or self.N2 % R != 0:
                raise ValueError(
                    "For equal-size random batches, both N1 and N2 must be divisible by random_batches. "
                    f"Here N1={self.N1}, N2={self.N2}, random_batches={R}."
                )

        if self.auto_B_from_charges_masses:
            q1sq = self.charge1 * self.charge1
            q2sq = self.charge2 * self.charge2
            # B_ji is the coefficient in the equation for target species i.
            # For physical multispecies Landau scaling, B_ji is proportional
            # to q_i^2*q_j^2/m_i, so m_i*B_ji = m_j*B_ij for cross pairs.
            self.B11 = q1sq * q1sq / self.m1
            self.B21 = q1sq * q2sq / self.m1
            self.B12 = q1sq * q2sq / self.m2
            self.B22 = q2sq * q2sq / self.m2

        self.x_grid = (np.arange(self.Nx, dtype=float) + 0.5) * self.eta
        self.x_faces = np.arange(self.Nx, dtype=float) * self.eta
        self.masses = np.array([self.m1, self.m2], dtype=float)
        self.charges = np.array([self.charge1, self.charge2], dtype=float)
        self.densities = np.array([self.n1, self.n2], dtype=float)
        self.B = np.array([[self.B11, self.B12], [self.B21, self.B22]], dtype=float)

    @property
    def ntot(self) -> int:
        return math.ceil(self.T / self.dt)

    @property
    def particles_per_x_cell_1(self) -> int:
        return self.Nv1_v1 * self.Nv1_v2 * self.Nc1

    @property
    def particles_per_x_cell_2(self) -> int:
        return self.Nv2_v1 * self.Nv2_v2 * self.Nc2

    @property
    def q_over_m1(self) -> float:
        return self.charge1 / self.m1

    @property
    def q_over_m2(self) -> float:
        return self.charge2 / self.m2

    @property
    def current_factor1(self) -> float:
        if self.current_model == "charge_over_mass":
            return self.charge1 / self.m1
        return self.charge1

    @property
    def current_factor2(self) -> float:
        if self.current_model == "charge_over_mass":
            return self.charge2 / self.m2
        return self.charge2

    @property
    def collision_active(self) -> bool:
        return self.enable_collisions and self.collision_strength != 0.0

    @property
    def gamma_landau_collisionless(self) -> float:
        # Electron-scale collisionless linear Landau-damping reference.
        return -(1.0 / self.k**3) * math.sqrt(math.pi / 8.0) * math.exp(-1.0 / (2.0 * self.k**2) - 1.5)
