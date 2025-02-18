import numpy as np


def ricker_time(t: np.ndarray, nu0: float) -> np.ndarray:
    """Calculate Ricker's impulse.

    Ricker's impulse is given by the formula
    f(t) = (1 - 2 * (nu0 * pi * t)^2) * exp(-(nu0 * pi * t)^2).

    :param t: Time array where to calculate impulse values.
    :type t: ndarray
    :param nu0: Main frequency of the impulse.
    :type nu0: float
    :return: Impulse values at given times.
    :rtype: ndarray"""
    om0 = nu0 * np.pi
    arg = om0 * t
    sqr_arg = arg * arg
    ans = (1.0 - 2.0 * sqr_arg) * np.exp(-sqr_arg)
    return ans


def ricker_der_time(t: np.ndarray, nu0: float) -> np.ndarray:
    """Calculate Ricker's impulse derivative.

    Ricker's impulse derivative is given by the formula
    f'(t) = 2 * (nu0 * pi)^2 * t * (2 * (nu0 * pi * t)^2 - 3) * exp(-(nu0 * pi * t)^2).
    We normalize it so the amplitude maximum is equal to 1.

    :param t: Time array where to calculate impulse values.
    :type t: ndarray
    :param nu0: Main frequency of the impulse.
    :type nu0: float
    :return: Impulse values at given times.
    :rtype: ndarray"""
    om0 = nu0 * np.pi
    arg = om0 * t
    sqr_arg = arg * arg
    ans = 2.0 * om0**2 * t * (2.0 * sqr_arg - 3.0) * np.exp(-sqr_arg)
    return ans / np.max(np.abs(ans))


def impulse_support(dt: float, t_supp: float) -> np.ndarray:
    """Calculate impulse support array.

    Support here is defined as interval (-t, t) where impulse may have nonzero values.

    :param dt: Time array where to calculate impulse values.
    :type dt: float
    :param t_supp: Interval (-t, t) border.
    :type t_supp: float
    :return: Impulse support array.
    :rtype: ndarray"""
    n = int(t_supp / dt)
    nt = 2 * n + 1
    ts = np.zeros(nt)
    for i in range(nt):
        ts[i] = float(i - n) * dt
    return ts


def convolution_matrix(impulse: np.ndarray, n: int) -> np.ndarray:
    """Create convolution matrix.

    This function creates square convolution matrix for given impulse in explicit form.

    :param impulse: Impulse to convolve with.
    :type impulse: ndarray
    :param n: Size of matrix.
    :type n: int
    :return: Convolution matrix W.
    :rtype: ndarray"""
    k = (len(impulse) - 1) // 2
    W = np.zeros((n, n), dtype=float)
    for j in range(n):
        for i in range(n):
            diff = j - i
            if abs(diff) <= k:
                W[i, j] = impulse[k - diff]
    return W
