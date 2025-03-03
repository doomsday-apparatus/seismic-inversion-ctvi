import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from scripts.utils import read_model


def _plot_impedance_2d(
    ox: np.ndarray,
    ot: np.ndarray,
    imp_true: np.ndarray,
    imp_init: np.ndarray,
    imp_rec: np.ndarray,
):
    # Plot settings
    plt.rc("font", size=16)
    plt.rc("axes", titlesize=18)
    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    # Create axes
    fig = plt.figure(figsize=(10, 9), layout="constrained")
    gs_base = GridSpec(2, 1, fig)
    gs0 = gs_base[0].subgridspec(1, 2)
    gs01 = gs_base[1].subgridspec(1, 3, width_ratios=[1, 2, 1])
    gs1 = gs01[1].subgridspec(1, 1)
    ax0 = fig.add_subplot(gs0[0])
    axt = fig.add_subplot(gs0[1])
    axr = fig.add_subplot(gs1[0])
    axes_model = [ax0, axt, axr]
    # Plot figures
    vmin, vmax = 1.0e-6 * np.min(imp_true), 1.0e-6 * np.max(imp_true)
    cmap = plt.colormaps["jet"]
    imt = axt.pcolormesh(
        ox,
        ot,
        1.0e-6 * imp_true,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    im0 = ax0.pcolormesh(
        ox,
        ot,
        1.0e-6 * imp_init,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    imr = axr.pcolormesh(
        ox,
        ot,
        1.0e-6 * imp_rec,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    # Plot colorbars
    cbar_t = plt.colorbar(imt)
    cbar_0 = plt.colorbar(im0)
    cbar_r = plt.colorbar(imr)
    cbars = [cbar_t, cbar_0, cbar_r]
    # Format colorbars
    for cb in cbars:
        cb.set_label("Impedance (km/s * g/cm3)")
        cb.ax.tick_params(labelsize=16)
    # Format axes
    titles = ["a) Initial", "b) True", "c) Reconstructed"]
    for ax, ti in zip(axes_model, titles):
        ax.set_title(ti)
        ax.set_xlim(0, 7)
        ax.invert_yaxis()
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Time (s)")
    # TODO
    # fig.savefig('./fig_2d_imp.pdf', bbox_inches='tight', dpi=400)
    plt.show()


def _plot_image_rpp_2d(
    ox: np.ndarray,
    ot: np.ndarray,
    img_noise: np.ndarray,
    img_rec: np.ndarray,
    rpp_rec: np.ndarray,
):
    mult = 0.2
    # Plot settings
    plt.rc("font", size=16)
    plt.rc("axes", titlesize=18)
    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    # Create axes
    fig = plt.figure(figsize=(10, 9), layout="constrained")
    gs_base = GridSpec(2, 1, fig)
    gs0 = gs_base[0].subgridspec(1, 2)
    gs01 = gs_base[1].subgridspec(1, 3, width_ratios=[1, 2, 1])
    gs1 = gs01[1].subgridspec(1, 1)
    axn = fig.add_subplot(gs0[0])
    axs = fig.add_subplot(gs0[1])
    axr = fig.add_subplot(gs1[0])
    axes_model = [axn, axs, axr]
    # Plot figures
    amax = np.max(np.abs(img_noise))
    vmin, vmax = -mult * amax, mult * amax
    cmap = plt.colormaps["binary"]
    imn = axn.pcolormesh(
        ox,
        ot,
        img_noise,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ims = axs.pcolormesh(
        ox,
        ot,
        img_rec,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    imr = axr.pcolormesh(
        ox,
        ot,
        rpp_rec,
        cmap=cmap,
        shading="gouraud",
        vmin=-mult * np.max(np.abs(rpp_rec)),
        vmax=mult * np.max(np.abs(rpp_rec)),
        rasterized=True,
    )
    # Plot colorbars
    cbar_n = plt.colorbar(imn)
    cbar_s = plt.colorbar(ims)
    cbar_r = plt.colorbar(imr)
    cbars = [cbar_n, cbar_s, cbar_r]
    # Format colorbars
    for cb in cbars:
        cb.set_label("Amplitude")
        cb.ax.tick_params(labelsize=16)
    # Format axes
    titles = ["a) Noisy image", "b) Reconstructed image", "c) Reflection coefficient"]
    for ax, ti in zip(axes_model, titles):
        ax.set_title(ti)
        ax.set_xlim(0, 7)
        ax.invert_yaxis()
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Time (s)")
    # fig.savefig('./fig_2d_img.pdf', bbox_inches='tight', dpi=400)
    plt.show()


def _plot_1d(
    ix: int,
    ot: np.ndarray,
    imp_true: np.ndarray,
    imp_init: np.ndarray,
    imp_rec: np.ndarray,
    img_noise: np.ndarray,
    img_rec: np.ndarray,
    rpp_rec: np.ndarray,
):
    # Plot settings
    plt.rc("font", size=16)
    plt.rc("axes", titlesize=18)
    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    lw1, lw2 = 4, 2
    # Create axes
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs = GridSpec(1, 2, fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    axes = [ax1, ax2]
    # Plot figures
    ax1.plot(1.0e-6 * imp_true[:, ix], ot, color="black", label="True", lw=lw1)
    ax1.plot(1.0e-6 * imp_init[:, ix], ot, color="blue", label="Initial", lw=lw1)
    ax1.plot(1.0e-6 * imp_rec[:, ix], ot, color="red", label="Reconstructed", lw=lw1)
    ax2.plot(img_noise[:, ix], ot, color="black", label="Noisy seismic track", lw=lw2)
    ax2.plot(
        img_rec[:, ix], ot, color="red", label="Reconstructed seismic track", lw=lw2
    )
    ax2.plot(rpp_rec[:, ix], ot, color="blue", label="Reflection coefficient", lw=lw2)
    # Format axes
    for ax in axes:
        ax.set_ylim(ot.min(), ot.max())
        ax.set_ylabel("Time (s)")
        ax.invert_yaxis()
        ax.grid()
        ax.legend()
    ax1.set_xlabel("Impedance (km/s * g/cm3)")
    ax2.set_xlabel("Amplitude")
    # fig.savefig('./fig_1d.pdf', bbox_inches='tight')
    plt.show()


def _main():
    nx, dx = 120, 60.0
    nt, dt = 782, 0.004
    ox = 1.0e-3 * np.linspace(0.0, dx * (nx - 1), nx)
    ot = np.linspace(0.0, dt * (nt - 1), nt)
    ix = 15
    path = Path(os.getcwd()) / "data"

    imp_true_time = read_model(path / "other" / f"imp_true_time_nt{nt:d}", nx, nt)
    imp_init_time = read_model(path / "input" / f"imp_init_time_nt{nt:d}", nx, nt)
    imp_rec_time = read_model(path / "output" / f"imp_rec_time_nt{nt:d}", nx, nt)

    img_time = read_model(path / "input" / f"img_synt_time_nt{nt:d}_noise30", nx, nt)
    img_rec_time = read_model(path / "output" / f"img_rec_time_nt{nt:d}", nx, nt)

    rpp_rec_time = read_model(path / "output" / f"rpp_rec_time_nt{nt:d}", nx, nt)

    _plot_impedance_2d(ox, ot, imp_true_time, imp_init_time, imp_rec_time)
    _plot_image_rpp_2d(ox, ot, img_time, img_rec_time, rpp_rec_time)
    _plot_1d(
        ix,
        ot,
        imp_true_time,
        imp_init_time,
        imp_rec_time,
        img_time,
        img_rec_time,
        rpp_rec_time,
    )


if __name__ == "__main__":
    _main()
