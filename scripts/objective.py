import numpy as np
from typing import Tuple
from .utils import apply_integration_matrix
from .utils import apply_integration_matrix_transpose


def objective_ctvi(
    x: np.ndarray,
    conv_mat: np.ndarray,
    seis_track: np.ndarray,
    tv_mat: np.ndarray,
    xi: np.ndarray,
    rpp_neigh: np.ndarray,
    alpha_ss: float,
    alpha_tv: float,
    alpha_m0: float,
    alpha_n: float,
) -> Tuple[float, np.ndarray]:
    """Cost function for constrained total variation inversion (CTVI).

    Cost function has the form
    f(r) = ||Wr - s||_1
        + a_{ss} * ||r||_1
        + a_{tv} * ||grad r||_1
        + a_{m0} * ||Cr - xi||_1
        + a_{n}  * ||C(r - r_{neigh})||_1.

    :param x: Current nonlinear iteration solution.
    :type x: ndarray
    :param conv_mat: Convolution matrix, direct problem operator.
    :type conv_mat: ndarray
    :param seis_track: Seismic track to invert, inverse problem data.
    :type seis_track: ndarray
    :param tv_mat: Total variation matrix to use corresponding stabilizer.
    :type tv_mat: ndarray
    :param xi: Vector of logarithm of impedance to use initial model stabilizer.
    :type xi: ndarray
    :param rpp_neigh: Neighboring reflection coefficient track
    to ensure spatial connectivily of the solution.
    :type rpp_neigh: ndarray
    :param alpha_ss: Sparse spike regularization parameter
    for ||r||_1 stabilizer.
    :type alpha_ss: float
    :param alpha_tv: Total variation regularization parameter
    for ||grad r||_1 stabilizer.
    :type alpha_tv: float
    :param alpha_m0: Initial model regularization parameter
    for ||Cr - xi||_1 stabilizer.
    :type alpha_m0: float
    :param alpha_n: Spatial connectivity regularization parameter
    for ||C(r - r_neigh)||_1 stabilizer.
    :type alpha_n: float
    :return: Value and gradient of the cost function on current model x.
    :rtype: Tuple[float, ndarray]"""
    delta_sqr = 1.0e-6**2
    rpp = x
    f = 0.0
    dfdm = np.zeros(x.size, dtype=float)
    # Residual ||Wr - s||_1
    residual = conv_mat.dot(rpp) - seis_track
    f += np.dot(residual, residual)
    dfdm += 2.0 * conv_mat.transpose().dot(residual)
    if alpha_ss > 0.0:
        # Stabilizer ||rpp||_1
        sqrt_delta_r = np.sqrt(rpp**2 + delta_sqr)
        f += alpha_ss * np.sum(sqrt_delta_r)
        dfdm += alpha_ss * (rpp / sqrt_delta_r)
    if alpha_tv > 0.0:
        # Stabilizer ||grad rpp||_1
        Tr = tv_mat.dot(rpp)
        sqrt_delta_Tr = np.sqrt(Tr**2 + delta_sqr)
        f += alpha_tv * np.sum(sqrt_delta_Tr)
        dfdm += alpha_tv * tv_mat.transpose().dot(Tr / sqrt_delta_Tr)
    if alpha_m0 > 0.0:
        # Stabilizer ||Cr - xi||_1
        Cr_m_xi = apply_integration_matrix(rpp) - xi
        sqrt_delta_V = np.sqrt(Cr_m_xi**2 + delta_sqr)
        f += alpha_m0 * np.sum(sqrt_delta_V)
        dfdm += alpha_m0 * apply_integration_matrix_transpose(Cr_m_xi / sqrt_delta_V)
    if alpha_n > 0.0:
        # Stabilizer ||C(r - r_neigh||_1
        C_r_m_rn = apply_integration_matrix(rpp - rpp_neigh)
        sqrt_delta_V = np.sqrt(C_r_m_rn**2 + delta_sqr)
        f += alpha_n * np.sum(sqrt_delta_V)
        dfdm += alpha_n * apply_integration_matrix_transpose(C_r_m_rn / sqrt_delta_V)
    # print(f'func val = {f:g}')
    return f, dfdm
