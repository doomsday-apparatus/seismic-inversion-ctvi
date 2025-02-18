import numpy as np
import scipy.sparse as sp


def read_model(filename: str, nx: int, n: int) -> np.ndarray:
    """Read binary file as 2D array.

    It is assumed that files are written in column order (Fortran style), single precision.

    :param filename: Name of binary file.
    :type filename: str
    :param nx: Number of counts in lateral direction.
    :type nx: int
    :param n: Number of counts along depth/time axis.
    :type n: int
    :return: Contents of binary file as 2D array.
    :rtype: ndarray"""
    with open(filename, "rb") as file:
        data = np.fromfile(file, dtype=np.float32, count=n * nx).reshape(
            (n, nx), order="F"
        )
    return data


def write_model(filename: str, data: np.ndarray):
    """Write 2D array as binary file.

    File will be written in column order (Fortran style), single precision.

    :param filename: Name of binary file.
    :type filename: str
    :param data: 2D array to write.
    :type data: ndarray"""
    # Transpose to ensure column order
    data.transpose().astype(np.float32).tofile(filename)


def total_variation_matrix(n: int) -> np.ndarray:
    """Create numerical gradient matrix.

    Square numerical gradient matrix to use in ||grad r|| term of cost function.

    :param n: Size of matrix.
    :type n: int
    :return: Numerical gradient matrix.
    :rtype: ndarray"""
    row = np.array([], dtype=int)
    col = np.array([], dtype=int)
    val = np.array([], dtype=float)
    for i in range(n - 1):
        row = np.append(row, [i, i])
        col = np.append(col, [i, i + 1])
        val = np.append(val, [-1.0, 1.0])
    tv_mat = sp.coo_array((val, (row, col)), shape=((n - 1), n)).tocsr()
    return tv_mat


def integration_matrix(n: int) -> np.ndarray:
    """Create numerical integration matrix.

    Square numerical integration matrix to use in ||Cr - xi|| term of cost function.

    :param n: Size of matrix.
    :type n: int
    :return: Numerical integration matrix.
    :rtype: ndarray"""
    int_mat = np.zeros(shape=(n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            int_mat[i, j] = 1.0
    return int_mat


def apply_integration_matrix(x: np.ndarray) -> np.ndarray:
    """Calculate result of application of the numerical integration matrix C to vector x.

    :param x: Vector to be multiplied by the integration matrix.
    :type x:
    :return: Multiplication result Cx.
    :rtype: ndarray"""
    return np.cumsum(x)


def apply_integration_matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Calculate result of application of the transposed integration matrix C^T to vector x.

    :param x: Vector to be multiplied by the transposed integration matrix.
    :type x:
    :return: Multiplication result C^T * x.
    :rtype: ndarray"""
    return np.cumsum(x[::-1])[::-1]


def compute_reflection_coeff(Ip: np.ndarray, Ip0: np.ndarray | float) -> np.ndarray:
    """Compute reflection coefficient using acoustic impedance model.

    Reflection coefficient in calculated using the formula r_{i} = (Ip_{i} - Ip_{i-1}) / (Ip_{i} + Ip_{i-1}).

    :param Ip: Acoustic impedance model.
    :type Ip:
    :param Ip0: Acoustic impedance value in upper layer.
    :type Ip0: ndarray | float
    :return: Reflection coefficient model.
    :rtype: ndarray"""
    if Ip.ndim == 1:
        n = Ip.size
        rpp = np.zeros(n)
        rpp[0] = (Ip[0] - Ip0) / (Ip[0] + Ip0)
        for i in range(1, n):
            rpp[i] = (Ip[i] - Ip[i - 1]) / (Ip[i] + Ip[i - 1])
    elif Ip.ndim == 2:
        n = Ip.shape[0]
        rpp = np.zeros(Ip.shape)
        rpp[0, :] = (Ip[0, :] - Ip0) / (Ip[0, :] + Ip0)
        for i in range(1, n):
            rpp[i, :] = (Ip[i, :] - Ip[i - 1, :]) / (Ip[i, :] + Ip[i - 1, :])
    return rpp


def compute_impedance(rpp: np.ndarray, Ip0: float) -> np.ndarray:
    """Compute acoustic impedance using reflection coefficients.

    Acoustic impedance in calculated using the formula Ip_{i} = Ip_{i-1} * (1 + r_{i}) / (1 - r_{i}).

    :param rpp: Reflection coefficient model.
    :type rpp:
    :param Ip0: Acoustic impedance in upper layer.
    :type Ip0: float
    :return: Acoustic impedance model.
    :rtype: ndarray"""
    n = len(rpp)
    Ip = np.zeros(n)
    Ip[0] = Ip0 * (1 + rpp[0]) / (1 - rpp[0])
    for i in range(1, n):
        Ip[i] = Ip[i - 1] * (1 + rpp[i]) / (1 - rpp[i])
    return Ip
