import numpy as np
from typing import Tuple
from scipy.optimize import minimize

from .utils import compute_impedance
from .objective import objective_ctvi as obj


def inversion_1d(
    imp_init: np.ndarray,
    img_track: np.ndarray,
    conv_mat: np.ndarray,
    tv_mat: np.ndarray,
    rpp_neigh: np.ndarray,
    alpha_ss: float,
    alpha_tv: float,
    alpha_m0: float,
    alpha_n: float,
    maxiter: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert seismic track and obtain inverse problem solution.

    Solves optimization problem obtaining reconstructed acoustic impedance model.
    Uses regularization techniques in cost function.

    :param imp_init: Initial acoustic impedance model.
    :type imp_init: ndarray
    :param img_track: Seismic track, data for inverse problem.
    :type img_track: ndarray
    :param conv_mat: Convolution matrix, direct problem operator.
    :type conv_mat: ndarray
    :param tv_mat: Total variation matrix to use corresponding stabilizer.
    :type tv_mat: ndarray
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
    :param maxiter: Maximum number of iterations in optimization.
    :type maxiter: int
    :return: Reconstructed impedance model, synthetic seismic track
    for found solution and reflection coefficient track.
    :rtype: Tuple[ndarray, ndarray, ndarray]"""
    num_vars = img_track.size
    xi = 0.5 * np.log(imp_init / imp_init[0])
    rpp0 = np.zeros(num_vars, dtype=float)
    result = minimize(
        fun=obj,
        x0=rpp0,
        method="L-BFGS-B",
        jac=True,
        args=(
            conv_mat,
            img_track,
            tv_mat,
            xi,
            rpp_neigh,
            alpha_ss,
            alpha_tv,
            alpha_m0,
            alpha_n,
        ),
        options={"maxiter": maxiter},
    )
    rpp_rec = result.x
    imp_rec = compute_impedance(rpp_rec, imp_init[0])
    img_synt = np.dot(conv_mat, rpp_rec)
    return imp_rec, img_synt, rpp_rec
