import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from scripts.utils import read_model
from scripts.utils import write_model
from scripts.utils import compute_reflection_coeff
from scripts.impulse import ricker_der_time
from scripts.impulse import impulse_support
from scripts.impulse import convolution_matrix
from scripts.depth_to_time import create_timegrid
from scripts.depth_to_time import model_to_time_uniform
from scripts.depth_to_time import data_to_depth_uniform


def _main():
    # Axes parameters
    nx, nz = 120, 234
    dz = 15.0
    dt = 0.004
    # Impulse parameters
    t_supp = 0.100
    nu0 = 30.0
    # Gaussian noise level in data
    np.random.seed(42)
    noiselvl = 0.30
    # Paths
    data_path = Path(os.getcwd()) / "data"
    input_path = data_path / "input"
    other_path = data_path / "other"

    # Read input data
    c_true_depth = read_model(other_path / "c_true_nz234nx284dz15dx60", nx, nz)
    rho_true_depth = read_model(other_path / "rho_true_nz234nx284dz15dx60", nx, nz)
    imp_true_depth = c_true_depth * rho_true_depth
    c_init_depth = read_model(other_path / "c0_start3_nz234nx284dz15dx60", nx, nz)

    # Prepare data for Depth to Time transform
    timegrid = create_timegrid(c_init_depth, dz)
    tmax = np.max(timegrid[-1, :])
    nt = int(tmax // dt) + 2
    t_uniform = np.linspace(0.0, (nt - 1) * dt, nt)
    # Apply Depth to Time transform to models and data
    imp_true_time = model_to_time_uniform(imp_true_depth, timegrid, t_uniform)

    # Construct matrices for objective function
    t_imp_supp = impulse_support(dt, t_supp)
    impulse = ricker_der_time(t_imp_supp, nu0)
    conv_mat = convolution_matrix(impulse, nt)
    img_synt_time = np.zeros((nt, nx), dtype=float)
    for ix in tqdm(range(nx)):
        s = np.dot(
            conv_mat,
            compute_reflection_coeff(imp_true_time[:, ix], imp_true_depth[0, ix]),
        )
        noise = np.random.normal(0.0, 1.0, nt)
        img_synt_time[:, ix] = s + noiselvl * noise / np.linalg.norm(
            noise
        ) * np.linalg.norm(s)
    # Save data
    write_model(input_path / f"img_synt_time_nt{nt:d}_noise30", img_synt_time)
    # # Time to Depth using same velocity model
    # img_synt_depth = data_to_depth_uniform(img_synt_time, timegrid, t_uniform)
    # write_model(input_path / "img_synt_depth_noise30", img_synt_depth)


if __name__ == "__main__":
    _main()
