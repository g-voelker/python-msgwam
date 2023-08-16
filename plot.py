import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def make_plots(ds: xr.Dataset, output_path: str) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(8, 8)

    hours = ds['time'] / 3600
    u = ds['u'] - ds['u'].isel(time=0)
    amax = abs(u).max()

    axes[0, 0].pcolormesh(
        hours,
        ds['grid'] / 1000,
        u.T,
        vmin=-amax,
        vmax=amax,
        cmap='RdBu_r',
        shading='nearest'
    )

    axes[0, 0].set_title('$\\Delta u$ (m s$^{-1}$)')
    axes[0, 0].set_xlabel('time (hours)')
    axes[0, 0].set_ylabel('height (km)')

    plot_ray_property(hours, ds['r'] / 1000, axes[0, 1])
    plot_ray_property(hours, ds['m'], axes[1, 0])
    plot_ray_property(hours, ds['dens'], axes[1, 1])

    axes[0, 1].set_title('ray $r$')
    axes[1, 0].set_title('ray $m$')
    axes[1, 1].set_title('ray dens')

    for ax in axes.flatten():
        ax.set_xlim(0, hours.max())

    axes[0, 0].set_ylim(0, 100)
    axes[0, 1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

def plot_ray_property(time: np.ndarray, data: np.ndarray, ax: plt.Axes) -> None:
    for traj in data.T:
        ax.plot(time, traj, color='k', alpha=0.2)

    ax.set_xlabel('time (hours)')

if __name__ == '__main__':
    data_path, output_path = sys.argv[1:]
    with xr.open_dataset(data_path) as ds:
        make_plots(ds, output_path)
