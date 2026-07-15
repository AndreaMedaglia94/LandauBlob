"""Compact centred cardinal B-splines of order 1, 2, and 3."""
from numba import njit


@njit(inline="always", fastmath=True)
def abs1(x):
    return x if x >= 0.0 else -x


@njit(inline="always", fastmath=True)
def sign1(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@njit(inline="always", fastmath=True)
def spline_value_derivative(u, order):
    """Return B(u), B'(u) for the centred B-spline of order 1, 2 or 3."""
    a = abs1(u)

    if order == 1:
        if a <= 1.0:
            return 1.0 - a, -sign1(u)
        return 0.0, 0.0

    if order == 2:
        if a <= 0.5:
            return 0.75 - u * u, -2.0 * u
        if a <= 1.5:
            t = 1.5 - a
            return 0.5 * t * t, -sign1(u) * t
        return 0.0, 0.0

    # order == 3
    if a <= 1.0:
        return (4.0 - 6.0 * u * u + 3.0 * a * a * a) / 6.0, (-12.0 * u + 9.0 * u * a) / 6.0
    if a <= 2.0:
        t = 2.0 - a
        return t * t * t / 6.0, -0.5 * t * t * sign1(u)
    return 0.0, 0.0


@njit(inline="always", fastmath=True)
def periodic_delta(a, b, L):
    """Shortest signed periodic displacement a-b on a domain of length L."""
    d = a - b
    half = 0.5 * L
    if d > half:
        d -= L
    elif d < -half:
        d += L
    return d
