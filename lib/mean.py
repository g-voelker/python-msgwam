from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import config

if TYPE_CHECKING:
    from .rays import RayCollection

class MeanFlow:
    def __init__(self) -> None:
        """Initialize a mean flow."""

        self.r_faces = np.linspace(0, config.grid_max, config.n_grid)
        self.r_centers = (self.r_faces[:-1] + self.r_faces[1:]) / 2
        self.dr = self.r_faces[1] - self.r_faces[0]

        self.rho = self.init_rho()
        self.u, self.v = self.init_uv()
        self.dp_dx, self.dp_dy = self.init_grad_p()

    def __add__(self, other: np.ndarray) -> MeanFlow:
        """
        Return a new MeanFlow object sharing the same background profiles
        (density and pressure gradients) but with velocities equal to those of
        this object added to the data in other. Definied to make writing
        time-stepping routines easier.
        """

        if other.shape != (2, len(self.r_centers)):
            raise ValueError('other does not have correct shape')

        output = copy(self)
        output.u = self.u + other[0]
        output.v = self.v + other[1]

        return output

    def init_rho(self) -> np.ndarray:
        if config.boussinesq:
            return config.rhobar0 * np.ones(self.r_centers.shape)

        return config.rhobar0 * np.exp(-self.r_centers / config.hh)

    def init_uv(self) -> tuple[np.ndarray, np.ndarray]:
        method = config.uv_init_method
        if method == 'sine_homogeneous':
            tanh = np.tanh((self.r_centers - config.r0) / config.sig_r) + 1
            sine = np.sin(2 * np.pi * self.r_centers / config.sig_r)

            u = 0.5 * config.u0 * tanh * sine
            v = np.zeros(u.shape)

            return u, v

        message = f'Unknown method for initializing mean flow: {method}'
        raise ValueError(message)

    def init_grad_p(self) -> np.ndarray:
        return np.vstack((
            self.rho * config.f0 * self.v,
            -self.rho * config.f0 * self.u
        ))

    def project(
        self,
        rays: RayCollection,
        data: np.ndarray,
        grid: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Project the values in data (which should be the same shape as the
        properties stored in rays) onto vertical regions bounded by grid. If
        grid is None, self.r_faces will be used.
        """

        if grid is None:
            grid = self.r_faces

        r_lo = rays.r - 0.5 * rays.dr
        r_hi = rays.r + 0.5 * rays.dr

        r_mins = np.maximum(r_lo, grid[:-1, None])
        r_maxs = np.minimum(r_hi, grid[1:, None])
        fracs = np.maximum(r_maxs - r_mins, 0) / (grid[1] - grid[0])

        return np.nansum(fracs * data, axis=1)

    def dmean_dt(self, rays: RayCollection) -> np.ndarray:
        """
        Calculate du_dt and dv_dt, including the pseudomomentum flux divergence
        contribution from ray volumes.
        """

        cg_r = rays.cg_r()
        volume = abs(rays.dk * rays.dl * rays.dm)
        data = cg_r * rays.dens * volume

        pmf = np.zeros((2, len(self.r_faces)))
        pmf[0, 1:-1] = self.project(rays, data * rays.k, self.r_centers)
        pmf[1, 1:-1] = self.project(rays, data * rays.l, self.r_centers)

        pmf[:, 0] = pmf[:, 1]
        pmf[:, -1] = pmf[:, -2]

        dpmf_dr = np.diff(pmf, axis=1) / self.dr
        du_dt = config.f0 * self.v - (self.dp_dx + dpmf_dr[0]) / self.rho
        dv_dt = -config.f0 * self.u - (self.dp_dy + dpmf_dr[1]) / self.rho

        return np.vstack((du_dt, dv_dt))
