import sys

import matplotlib.pyplot as plt
import xarray as xr

def plot_wind(ds: xr.Dataset, output_path: str) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)

    u = ds['u']
    amax = abs(u).max()
    days = ds['time'] / (3600 * 24)
    grid = ds['grid'] / 1000

    ax.pcolormesh(
        days,
        grid,
        u.T,
        vmin=-amax,
        vmax=amax,
        cmap='RdBu_r',
        shading='nearest'
    )

    ax.set_title('$u$ (m s$^{-1}$)')
    ax.set_xlabel('time (hours)')
    ax.set_ylabel('height (km)')

    ax.set_xlim(0, days.max())
    ax.set_ylim(grid.min(), grid.max())

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)

if __name__ == '__main__':
    data_path, output_path = sys.argv[1:]
    with xr.open_dataset(data_path) as ds:
        plot_wind(ds, output_path)
