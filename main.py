import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from scripts.utils import read_model
from scripts.utils import write_model
from scripts.utils import total_variation_matrix
from scripts.impulse import ricker_der_time
from scripts.impulse import impulse_support
from scripts.impulse import convolution_matrix
from scripts.inversion import inversion_1d


def _main():
    # Axes parameters
    nx, nt = 120, 782
    dt = 0.004
    # Impulse parameters
    t_supp = 0.100
    nu0 = 30.0
    # Inversion parameters
    a_sparse_spike = 0.01
    a_total_var = 0.01
    a_init_model = 0.0001
    a_spatial_connect = 0.0001
    maxiter = 1000
    # Paths
    data_path = Path(os.getcwd()) / "data"
    input_path = data_path / "input"
    output_path = data_path / "output"

    # Read input data
    imp_init_time = read_model(input_path / f"imp_init_time_nt{nt:d}", nx, nt)
    img_time = read_model(input_path / f"img_synt_time_nt{nt:d}_noise30", nx, nt)

    # Construct matrices for objective function
    t_imp_supp = impulse_support(dt, t_supp)
    impulse = ricker_der_time(t_imp_supp, nu0)
    conv_mat = convolution_matrix(impulse, nt)
    tv_mat = total_variation_matrix(nt)
    a_neigh = 0.0
    rpp_neigh = np.zeros(nt, dtype=float)
    imp_rec_time = np.ones((nt, nx), dtype=float)
    img_rec_time = np.zeros((nt, nx), dtype=float)
    rpp_rec_time = np.zeros((nt, nx), dtype=float)
    # Apply seismic inversion to each track
    for ix in tqdm(range(nx), desc="Inversion of each seismic track"):
        if ix > 0:
            a_neigh = a_spatial_connect
        imp_rec_time[:, ix], img_rec_time[:, ix], rpp_rec_time[:, ix] = inversion_1d(
            imp_init_time[:, ix],
            img_time[:, ix],
            conv_mat,
            tv_mat,
            rpp_neigh,
            a_sparse_spike,
            a_total_var,
            a_init_model,
            a_neigh,
            maxiter,
        )
        rpp_neigh = rpp_rec_time[:, ix]
    # Save data
    write_model(output_path / f"imp_rec_time_nt{nt:d}", imp_rec_time)
    write_model(output_path / f"img_rec_time_nt{nt:d}", img_rec_time)
    write_model(output_path / f"rpp_rec_time_nt{nt:d}", rpp_rec_time)


if __name__ == "__main__":
    _main()
