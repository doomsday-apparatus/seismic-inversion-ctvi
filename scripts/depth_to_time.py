from enum import Enum
import numpy as np
from scipy.interpolate import CubicSpline


class _Interp(Enum):
    LINEAR = 1
    SPLINE = 3


class _Extrap(Enum):
    ZERO = 0
    CONSTANT = 1


def create_timegrid(vel: np.ndarray, dz: float) -> np.ndarray:
    """Calculate times using velocity model in depth domain.

    Time can be obtained using formula t(z) = 2 * int_{0}^{z} dw / c(x, w),
    where c(x, z) is the velocity in the depth domain.

    :param vel: Velocity c(x, z) in depth domain.
    :type vel: ndarray
    :param dz: Step size along depth oz axis.
    :type dz: float
    :return: Timegrid with corresponding time values.
    :rtype: ndarray"""
    nz, nx = vel.shape
    timegrid = np.zeros((nz, nx))
    oc = 1.0 / vel
    for iz in range(1, nz):
        timegrid[iz, :] = timegrid[iz - 1, :] + (oc[iz, :] + oc[iz - 1])
    timegrid *= dz
    return timegrid


def _to_time_uniform(
    model: np.ndarray,
    timegrid: np.ndarray,
    t_uni: np.ndarray,
    extrap: _Extrap,
    interp: _Interp,
) -> np.ndarray:
    _, nx = model.shape
    model_time_uni = np.zeros((t_uni.size, nx))
    for ix in range(nx):
        if interp == _Interp.LINEAR:
            if extrap == _Extrap.ZERO:
                model_time_uni[:, ix] = np.interp(
                    t_uni, timegrid[:, ix], model[:, ix], left=0.0, right=0.0
                )
            elif extrap == _Extrap.CONSTANT:
                model_time_uni[:, ix] = np.interp(t_uni, timegrid[:, ix], model[:, ix])
        else:
            cs = CubicSpline(timegrid[:, ix], model[:, ix])
            model_time_uni[:, ix] = cs(t_uni)
            if extrap == _Extrap.ZERO:
                model_time_uni[np.where(t_uni > timegrid[-1, ix])] = 0.0
            elif extrap == _Extrap.CONSTANT:
                model_time_uni[np.where(t_uni > timegrid[-1, ix])] = model[-1, ix]
    return model_time_uni


def model_to_time_uniform(
    model: np.ndarray, timegrid: np.ndarray, t_uni: np.ndarray
) -> np.ndarray:
    """Transform model from depth domain to time domain.

    :param model: Model in depth domain.
    :type model: ndarray
    :param timegrid: Grid with times corresponding to velocity model.
    :type timegrid: ndarray
    :param t_uni: Uniform grid in time domain.
    :type t_uni: ndarray
    :return: Interpolated model in time domain on uniform time grid.
    :rtype: ndarray"""
    return _to_time_uniform(model, timegrid, t_uni, _Extrap.CONSTANT, _Interp.LINEAR)


def data_to_time_uniform(
    data: np.ndarray, timegrid: np.ndarray, t_uni: np.ndarray
) -> np.ndarray:
    """Transform data from depth domain to time domain.

    :param data: Data in depth domain.
    :type data: ndarray
    :param timegrid: Grid with times corresponding to velocity model.
    :type timegrid: ndarray
    :param t_uni: Uniform grid in time domain.
    :type t_uni: ndarray
    :return: Interpolated data in time domain on uniform time grid.
    :rtype: ndarray"""
    return _to_time_uniform(data, timegrid, t_uni, _Extrap.ZERO, _Interp.SPLINE)


def _to_depth_uniform(
    model: np.ndarray,
    timegrid: np.ndarray,
    t_uni: np.ndarray,
    extrap: _Extrap,
    interp: _Interp,
) -> np.ndarray:
    _, nx = model.shape
    model_depth_uni = np.zeros(timegrid.shape)
    for ix in range(nx):
        if interp == _Interp.LINEAR:
            if extrap == _Extrap.ZERO:
                model_depth_uni[:, ix] = np.interp(
                    timegrid[:, ix], t_uni, model[:, ix], left=0.0, right=0.0
                )
            elif extrap == _Extrap.CONSTANT:
                model_depth_uni[:, ix] = np.interp(timegrid[:, ix], t_uni, model[:, ix])
        else:
            cs = CubicSpline(t_uni, model[:, ix])
            model_depth_uni[:, ix] = cs(timegrid[:, ix])
        # Process points outside of interval
        if extrap == _Extrap.ZERO:
            model_depth_uni[np.where(timegrid[:, ix] > t_uni[-1])] = 0.0
        elif extrap == _Extrap.CONSTANT:
            model_depth_uni[np.where(timegrid[:, ix] > t_uni[-1])] = model[-1, ix]
    return model_depth_uni


def model_to_depth_uniform(
    model: np.ndarray, timegrid: np.ndarray, t_uni: np.ndarray
) -> np.ndarray:
    """Transform model from time domain to depth domain.

    :param model: Model in time domain.
    :type model: ndarray
    :param timegrid: Grid with times corresponding to velocity model.
    :type timegrid: ndarray
    :param t_uni: Uniform grid in time domain.
    :type t_uni: ndarray
    :return: Interpolated model in depth domain on uniform depth grid.
    :rtype: ndarray"""
    return _to_depth_uniform(model, timegrid, t_uni, _Extrap.CONSTANT, _Interp.LINEAR)


def data_to_depth_uniform(
    model: np.ndarray, timegrid: np.ndarray, t_uni: np.ndarray
) -> np.ndarray:
    """Transform data from time domain to depth domain.

    :param data: Data in time domain.
    :type data: ndarray
    :param timegrid: Grid with times corresponding to velocity model.
    :type timegrid: ndarray
    :param t_uni: Uniform grid in time domain.
    :type t_uni: ndarray
    :return: Interpolated data in depth domain on uniform depth grid.
    :rtype: ndarray"""
    return _to_depth_uniform(model, timegrid, t_uni, _Extrap.ZERO, _Interp.SPLINE)
