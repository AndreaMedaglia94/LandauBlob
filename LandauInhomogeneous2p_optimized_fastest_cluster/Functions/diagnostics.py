"""Diagnostics and conservative collision rescaling for two-species 1D-2V C-PIC."""
import numpy as np


def species_number(w):
    return float(np.sum(w))


def kinetic_energy(v11, v12, w1, v21, v22, w2, par):
    return 0.5 * par.m1 * np.sum(w1 * (v11 * v11 + v12 * v12)) + \
           0.5 * par.m2 * np.sum(w2 * (v21 * v21 + v22 * v22))


def field_energy(E1, E2, B3, par):
    return 0.5 * par.eta * (np.sum(E1 * E1) + np.sum(E2 * E2) + np.sum(B3 * B3))


def total_mass_weighted_momentum(v11, v12, w1, v21, v22, w2, par):
    return np.array([
        par.m1 * np.sum(w1 * v11) + par.m2 * np.sum(w2 * v21),
        par.m1 * np.sum(w1 * v12) + par.m2 * np.sum(w2 * v22),
    ])


def total_charge_current(v11, v12, w1, v21, v22, w2, par):
    return np.array([
        par.current_factor1 * np.sum(w1 * v11) + par.current_factor2 * np.sum(w2 * v21),
        par.current_factor1 * np.sum(w1 * v12) + par.current_factor2 * np.sum(w2 * v22),
    ])


def species_bulk_temperature(v1, v2, w, mass):
    n = np.sum(w)
    u1 = np.sum(w * v1) / n
    u2 = np.sum(w * v2) / n
    T = mass * np.sum(w * ((v1 - u1) ** 2 + (v2 - u2) ** 2)) / (2.0 * n)
    return np.array([u1, u2]), float(T)


def total_bulk_temperature(v11, v12, w1, v21, v22, w2, par):
    n1 = np.sum(w1)
    n2 = np.sum(w2)
    M = par.m1 * n1 + par.m2 * n2
    P = total_mass_weighted_momentum(v11, v12, w1, v21, v22, w2, par)
    U = P / M
    thermal = 0.5 * par.m1 * np.sum(w1 * ((v11 - U[0]) ** 2 + (v12 - U[1]) ** 2)) + \
              0.5 * par.m2 * np.sum(w2 * ((v21 - U[0]) ** 2 + (v22 - U[1]) ** 2))
    T = thermal / (n1 + n2)
    return U, float(T), float(thermal)


def electric_l2(E1, par):
    return np.sqrt(par.eta * np.sum(E1 * E1))


def regularized_entropy_from_ftilde(ft1, ft2, w1, w2):
    if ft1 is None or ft2 is None:
        return np.nan
    s1 = -np.sum(w1 * np.log(np.maximum(ft1, 1.0e-300)))
    s2 = -np.sum(w2 * np.log(np.maximum(ft2, 1.0e-300)))
    return float(s1 + s2)


def _mom_energy(v11, v12, w1, v21, v22, w2, par):
    P = total_mass_weighted_momentum(v11, v12, w1, v21, v22, w2, par)
    K = kinetic_energy(v11, v12, w1, v21, v22, w2, par)
    return P, K


def conservative_rescale_to_target(v11, v12, v21, v22, w1, w2, par, target_P, target_K):
    """
    Restore total mass-weighted momentum and kinetic energy after the collision
    part of an explicit step.

    The transformation is common to both species:
        v <- U_target + a*(v - U_current).
    """
    n1 = np.sum(w1)
    n2 = np.sum(w2)
    M = par.m1 * n1 + par.m2 * n2

    Pcur, Kcur = _mom_energy(v11, v12, w1, v21, v22, w2, par)
    Ucur = Pcur / M
    Utar = target_P / M

    thermal_cur = Kcur - 0.5 * M * np.dot(Ucur, Ucur)
    thermal_tar = target_K - 0.5 * M * np.dot(Utar, Utar)
    if thermal_cur <= 0.0 or thermal_tar < 0.0:
        return v11, v12, v21, v22
    scale = np.sqrt(thermal_tar / thermal_cur) if thermal_tar > 0.0 else 0.0

    v11n = Utar[0] + scale * (v11 - Ucur[0])
    v12n = Utar[1] + scale * (v12 - Ucur[1])
    v21n = Utar[0] + scale * (v21 - Ucur[0])
    v22n = Utar[1] + scale * (v22 - Ucur[1])
    return v11n, v12n, v21n, v22n


class History:
    def __init__(self):
        self.t = []
        self.E1_l2 = []
        self.kinetic = []
        self.field = []
        self.total_energy = []
        self.mom1 = []
        self.mom2 = []
        self.current1 = []
        self.current2 = []
        self.U1_v1 = []
        self.U1_v2 = []
        self.U2_v1 = []
        self.U2_v2 = []
        self.T1 = []
        self.T2 = []
        self.Ttot = []
        self.entropy = []

    def append(self, t, x1, v11, v12, w1, x2, v21, v22, w2, E1, E2, B3, par, ft1=None, ft2=None):
        ke = kinetic_energy(v11, v12, w1, v21, v22, w2, par)
        fe = field_energy(E1, E2, B3, par)
        mom = total_mass_weighted_momentum(v11, v12, w1, v21, v22, w2, par)
        cur = total_charge_current(v11, v12, w1, v21, v22, w2, par)
        U1, T1 = species_bulk_temperature(v11, v12, w1, par.m1)
        U2, T2 = species_bulk_temperature(v21, v22, w2, par.m2)
        _, Ttot, _ = total_bulk_temperature(v11, v12, w1, v21, v22, w2, par)
        self.t.append(float(t))
        self.E1_l2.append(float(electric_l2(E1, par)))
        self.kinetic.append(float(ke))
        self.field.append(float(fe))
        self.total_energy.append(float(ke + fe))
        self.mom1.append(float(mom[0]))
        self.mom2.append(float(mom[1]))
        self.current1.append(float(cur[0]))
        self.current2.append(float(cur[1]))
        self.U1_v1.append(float(U1[0]))
        self.U1_v2.append(float(U1[1]))
        self.U2_v1.append(float(U2[0]))
        self.U2_v2.append(float(U2[1]))
        self.T1.append(float(T1))
        self.T2.append(float(T2))
        self.Ttot.append(float(Ttot))
        self.entropy.append(float(regularized_entropy_from_ftilde(ft1, ft2, w1, w2)))

    def as_dict(self):
        names = (
            "t", "E1_l2", "kinetic", "field", "total_energy", "mom1", "mom2",
            "current1", "current2", "U1_v1", "U1_v2", "U2_v1", "U2_v2",
            "T1", "T2", "Ttot", "entropy"
        )
        return {name: np.asarray(getattr(self, name)) for name in names}


class EnergyLog:
    """
    Lightweight energy diagnostic recorded at *every* time step (unlike
    History, which is recorded only every par.record_interval and carries
    the heavier per-species diagnostics such as temperature and entropy).

    Both field_energy and kinetic_energy are O(Nx) / O(N) sums, so logging
    them every step is cheap relative to the collision/deposition work.
    """
    def __init__(self):
        self.t = []
        self.E_field = []
        self.E_kinetic = []
        self.E_total = []

    def append(self, t, v11, v12, w1, v21, v22, w2, E1, E2, B3, par):
        fe = field_energy(E1, E2, B3, par)
        ke = kinetic_energy(v11, v12, w1, v21, v22, w2, par)
        self.t.append(float(t))
        self.E_field.append(float(fe))
        self.E_kinetic.append(float(ke))
        self.E_total.append(float(fe + ke))

    def as_dict(self):
        names = ("t", "E_field", "E_kinetic", "E_total")
        return {name: np.asarray(getattr(self, name)) for name in names}
