import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

def animate(ds: xr.Dataset, output_path: str) -> None:
    fig, axes = plt.subplots(ncols=3)
    fig.set_size_inches(12, 4)

    z = ds['grid'].values / 1000
    u = ds['u'].values

    m = ds['m'].values
    r = ds['r'].values / 1000
    dens = ds['dens'].values
    pmf = ds['pmf_u'].values * 1000

    u_max = 1.2 * abs(u).max()
    m_min = 1.2 * np.nanmin(m)
    m_max = 0.8 * np.nanmax(m)
    pmf_max = 1.2 * abs(pmf).max()
    
    axes[0].set_xlim(-u_max, u_max)
    axes[0].set_ylim(0, 100)

    axes[1].set_xlim(-pmf_max, pmf_max)
    axes[1].set_ylim(0, 100)

    axes[2].set_xlim(m_min, m_max)
    axes[2].set_ylim(0, 100)

    axes[0].set_xlabel('$u$ (m s$^{-1}$)')
    axes[0].set_ylabel('$z$ (km)')
    axes[0].set_title('mean wind')

    axes[1].set_xlabel('$c_\mathrm{g} k\mathcal{A}$ (mPa)')
    axes[1].set_ylabel('$z$ (km)')
    axes[1].set_title('pseudomomentum flux')

    axes[2].set_xlabel('$m$ (m$^{-1}$)')
    axes[2].set_ylabel('$z$ (km)')
    axes[2].set_title('ray volumes')

    line_u, = axes[0].plot([], [], color='k')
    line_pmf, = axes[1].plot([], [], color='k')
    norm = LogNorm(np.nanmin(dens), np.nanmax(dens))
    dots = axes[2].scatter([], [], 30, c=[], alpha=0.3, cmap='Greys', norm=norm)

    def update(i: int) -> None:
        line_u.set_data(u[i], z)
        line_pmf.set_data(pmf[i], z)
        dots.set_offsets(np.vstack((m[i], r[i])).T)
        dots.set_array(dens[i])

        hours = i / 30
        days = int(hours / 24)
        hours = hours - 24 * days

        fig.suptitle(f'{days} days, {hours:.2f} hours')
        plt.tight_layout()

        return line_u, line_pmf, dots

    ani = FuncAnimation(fig, update, frames=len(u), interval=15, blit=True)
    ani.save(output_path)

if __name__ == '__main__':
    data_path, output_path = sys.argv[1:]
    with xr.open_dataset(data_path) as ds:
        animate(ds, output_path)