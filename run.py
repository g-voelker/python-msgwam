import sys

import matplotlib.pyplot as plt
import numpy as np

import lib.config as config

from lib.config import load_config
from lib.solver import Solver

if __name__ == '__main__':
    path = sys.argv[1]
    load_config(path)

    solver = Solver()
    solver.integrate()

    flux = np.zeros((solver.nt_max, *solver.mean.r_faces.shape))
    for i in range(solver.nt_max):
        solver.rays.data = solver.int_rays[i]
        vol = config.dk_init * config.dl_init * solver.rays.dm
        cg_r = solver.rays.cg_r()

        data = cg_r * solver.rays.dens * vol
        proj = solver.mean.project(solver.rays, data, solver.mean.r_centers)
        flux[i, 1:-1] = proj

    flux[:, 0] = flux[:, 1]
    flux[:, -1] = flux[:, -2]
    flux_div = np.diff(flux, axis=1) / solver.mean.dr

    time = np.arange(solver.nt_max)
    r = solver.mean.r_centers
    amax = abs(flux_div).max()
    
    fig, ax = plt.subplots()
    ax.contourf(time, r, flux_div.T, cmap='RdBu_r', vmin=-amax, vmax=amax)
    plt.show()