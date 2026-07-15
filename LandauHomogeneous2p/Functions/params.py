"""
Parameter object for the two-species homogeneous Landau particle code.

This file is the Python analogue of the MATLAB SetParam.m file.  Edit
Params(...) in the main script, or change the defaults below, to choose
masses, domains, collision coefficients, time step, mollifier, and the
optional conservative post-collision rescaling.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math
import numpy as np


@dataclass
class Params:
    # --- collision kernel A_ji(z) = B_ji |z|^gam (|z|^2 I - z otimes z) ---
    gam: float = 0.0

    # --- particles / velocity domains ---
    Nv: int = 30
    Npc: int = 1
    Lv1: float = 4.0
    Lv2: float = 4.0
    auto_Lv2_equal_m_epsi: bool = True

    # --- physical parameters ---
    m1: float = 1.0
    m2: float = 2.0
    n1: float = 1.0
    n2: float = 1.0

    # Matrix B_ji: B21 acts on species 1 by particles of species 2, B12 vice versa.
    B11: float = 1.0 / 32.0
    B21: float = 1.0 / 16.0
    B12: float = 1.0 / 16.0
    B22: float = 1.0 / 8.0

    # --- BKW exact solution constants ---
    beta: float = 1.0 / 16.0
    c: float = 0.5
    temp: float = 1.0

    # --- Gaussian/Maxwellian initial data parameters for non-BKW tests ---
    temp_1_0: float = 1.0
    temp_2_0: float = 1.0
    u_1_x_0: float = 0.0
    u_1_y_0: float = 0.0
    u_2_x_0: float = 0.0
    u_2_y_0: float = 0.0

    # --- time stepping ---
    T: float = 5.0
    dt: float = 0.01

    # --- mollifier: 0=Gaussian, 1/2/3=tensor B-spline order ---
    shape: int = 3

    # --- optional MATLAB-style conservative correction after each Euler step ---
    conservative_rescaling: bool = True

    # --- initialization details ---
    random_seed: Optional[int] = 12345
    enforce_initial_moments: bool = True

    # --- performance ---
    block_size: int = 1000

    # --- derived quantities -----------------------------------------
    N: int = field(init=False, repr=False)
    n: float = field(init=False, repr=False)
    rho: float = field(init=False, repr=False)

    dV1: float = field(init=False, repr=False)
    dV2: float = field(init=False, repr=False)
    h1: float = field(init=False, repr=False)
    h2: float = field(init=False, repr=False)
    epsi1: float = field(init=False, repr=False)
    epsi2: float = field(init=False, repr=False)

    Vc1: np.ndarray = field(init=False, repr=False)
    Vc2: np.ndarray = field(init=False, repr=False)
    grid_x1: np.ndarray = field(init=False, repr=False)
    grid_y1: np.ndarray = field(init=False, repr=False)
    grid_x2: np.ndarray = field(init=False, repr=False)
    grid_y2: np.ndarray = field(init=False, repr=False)

    B: np.ndarray = field(init=False, repr=False)
    masses: np.ndarray = field(init=False, repr=False)
    densities: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.shape not in (0, 1, 2, 3):
            raise ValueError("shape must be 0, 1, 2, or 3")
        if self.Nv <= 0 or self.Npc <= 0:
            raise ValueError("Nv and Npc must be positive")
        if self.dt <= 0.0 or self.T < 0.0:
            raise ValueError("dt must be positive and T must be non-negative")
        if self.m1 <= 0.0 or self.m2 <= 0.0:
            raise ValueError("species masses must be positive")
        if self.n1 <= 0.0 or self.n2 <= 0.0:
            raise ValueError("species number densities must be positive")

        if self.auto_Lv2_equal_m_epsi:
            self.Lv2 = self.Lv1 * (self.m1 / self.m2) ** (1.0 / 1.98)
        if self.Lv1 <= 0.0 or self.Lv2 <= 0.0:
            raise ValueError("Lv1 and Lv2 must be positive")

        self.N = self.Nv**2 * self.Npc
        self.n = self.n1 + self.n2
        self.rho = self.m1 * self.n1 + self.m2 * self.n2
        self.masses = np.array([self.m1, self.m2], dtype=float)
        self.densities = np.array([self.n1, self.n2], dtype=float)
        self.B = np.array([[self.B11, self.B12], [self.B21, self.B22]], dtype=float)

        self.Vc1, self.dV1, self.h1, self.grid_x1, self.grid_y1 = self._make_grid(self.Lv1)
        self.Vc2, self.dV2, self.h2, self.grid_x2, self.grid_y2 = self._make_grid(self.Lv2)

        if self.shape == 0:
            self.epsi1 = 0.64 * self.h1**1.98
            self.epsi2 = 0.64 * self.h2**1.98
        else:
            self.epsi1 = self.h1
            self.epsi2 = self.h2

    def _make_grid(self, Lv: float) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        edges = np.linspace(-Lv, Lv, self.Nv + 1)
        dV = edges[1] - edges[0]
        centers = edges[:-1] + 0.5 * dV
        h = 2.0 * Lv / self.Nv
        GX, GY = np.meshgrid(centers, centers, indexing="ij")
        return centers, dV, h, GX.ravel(), GY.ravel()

    @property
    def ntot(self) -> int:
        return math.ceil(self.T / self.dt)

    def species_grid(self, species: int):
        """Return (grid_x, grid_y, dV, h, epsi, mass, density) for species 1 or 2."""
        if species == 1:
            return self.grid_x1, self.grid_y1, self.dV1, self.h1, self.epsi1, self.m1, self.n1
        if species == 2:
            return self.grid_x2, self.grid_y2, self.dV2, self.h2, self.epsi2, self.m2, self.n2
        raise ValueError("species must be 1 or 2")
