"""Diagnostics for the 1D-2V C-PIC Landau damping run."""
import numpy as np


def kinetic_energy(v1, v2, w):
    return 0.5 * np.sum(w * (v1 * v1 + v2 * v2))


def field_energy(E1, E2, B3, par):
    return 0.5 * par.eta * (np.sum(E1 * E1) + np.sum(E2 * E2) + np.sum(B3 * B3))


def total_momentum(v1, v2, w):
    return np.array([np.sum(w * v1), np.sum(w * v2)])


def electric_l2(E1, par):
    return np.sqrt(par.eta * np.sum(E1 * E1))


def regularized_entropy_from_ftilde(ft, w):
    """S_eta,eps[f^N] = -sum_p w_p log(f_tilde(x_p,v_p))."""
    if ft is None:
        return np.nan
    return -np.sum(w * np.log(np.maximum(ft, 1.0e-300)))


class History:
    def __init__(self):
        self.t = []
        self.E1_l2 = []
        self.kinetic = []
        self.field = []
        self.total_energy = []
        self.mom1 = []
        self.mom2 = []
        self.entropy = []

    def append(self, t, x, v1, v2, w, E1, E2, B3, par, ft=None):
        ke = kinetic_energy(v1, v2, w)
        fe = field_energy(E1, E2, B3, par)
        mom = total_momentum(v1, v2, w)
        self.t.append(float(t))
        self.E1_l2.append(float(electric_l2(E1, par)))
        self.kinetic.append(float(ke))
        self.field.append(float(fe))
        self.total_energy.append(float(ke + fe))
        self.mom1.append(float(mom[0]))
        self.mom2.append(float(mom[1]))
        self.entropy.append(float(regularized_entropy_from_ftilde(ft, w)))

    def as_dict(self):
        return {name: np.asarray(getattr(self, name)) for name in (
            "t", "E1_l2", "kinetic", "field", "total_energy", "mom1", "mom2", "entropy"
        )}
