from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import config

if TYPE_CHECKING:
    from .rays import RayCollection

class MeanFlow:
    def __init__(self) -> None:
        """
        Initialize a MeanFlow using the configuration settings. Note that
        config.load_config must have been called for initialization to succeed.
        """

        self.r_faces = np.linspace(*config.grid_bounds, config.n_grid)
        self.r_centers = (self.r_faces[:-1] + self.r_faces[1:]) / 2
        self.dr = self.r_faces[1] - self.r_faces[0]

        self.rho = self.init_rho()
        self.u, self.v = self.init_uv()
        self.dp_dx, self.dp_dy = self.init_grad_p()

    def __add__(self, other: np.ndarray) -> MeanFlow:
        """
        Return a new MeanFlow object sharing the same background profiles
        (density and pressure gradients) but with velocities equal to those of
        this object added to the data in other. Defined to make writing
        time-stepping routines easier.

        Parameters
        ----------
        other
            Array of data to be added to the wind profiles (for example, the
            calculated mean flow tendency multiplied by the time step).

        Returns
        -------
        MeanFlow
            MeanFlow with updated wind profiles.

        Raises
        ------
        ValueError
            Indicates that other does not have the correct shape.

        """

        if other.shape != (2, len(self.r_centers)):
            raise ValueError('other does not have correct shape')

        output = copy(self)
        output.u = self.u + other[0]
        output.v = self.v + other[1]

        return output

    def init_rho(self) -> np.ndarray:
        """
        Initialize the mean density profile depending on whether or not the
        Boussinesq approximation is made.

        Returns
        -------
        np.ndarray
            Mean density profile at cell centers.

        """

        if config.boussinesq:
            return config.rhobar0 * np.ones(self.r_centers.shape)

        return config.rhobar0 * np.exp(-self.r_centers / config.hh)

    def init_uv(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize the mean wind profiles using the method identified in config.

        Returns
        -------
        np.ndarray, np.ndarray
            Mean u and v at cell centers.

        Raises
        ------
        ValueError
            Indicates that an unknown method was specified for initialization.
            Currently, the only method implemented is sine_homogeneous.

        """

        method = config.uv_init_method
        if method == 'sine_homogeneous':
            tanh = np.tanh((self.r_centers - config.r0) / config.sig_r) + 1
            sine = np.sin(2 * np.pi * self.r_centers / config.sig_r)

            u = 0.5 * config.u0 * tanh * sine
            v = np.zeros(u.shape)

            return u, v

        if method == 'gaussian':
            arg = -(((self.r_centers - config.r0) / config.sig_r) ** 2)
            u = config.u0 * np.exp(arg)
            v = np.zeros(u.shape)

            return u, v

        message = f'Unknown method for initializing mean flow: {method}'
        raise ValueError(message)

    def init_grad_p(self) -> np.ndarray:
        """
        Initialize the horizontal pressure gradients according to the
        geostrophic approximation.

        Returns
        -------
        np.ndarray
            Array whose first and second rows correspond to the zonal and
            meridional pressure gradients, respectively, at cell centers.

        """

        return np.vstack((
            self.rho * config.f0 * self.v,
            -self.rho * config.f0 * self.u
        ))

    def get_fracs(
        self,
        rays: RayCollection,
        grid: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Find the fraction of each grid cell that intersects each ray volume.

        Parameters
        ----------
        rays
            Current ray volume properties.
        grid, optional
            The edges of the vertical regions to project onto. If None, then the
            cell faces of the mean grid are used, such that the vertical regions
            are the cells themselves.

        Returns
        -------
        np.ndarray
            Fraction of each grid cell intersected by each ray. Has shape
            (len(grid - 1), config.n_ray_max).

        """

        if grid is None:
            grid = self.r_faces

        r_lo = rays.r - 0.5 * rays.dr
        r_hi = rays.r + 0.5 * rays.dr

        r_mins = np.maximum(r_lo, grid[:-1, None])
        r_maxs = np.minimum(r_hi, grid[1:, None])

        return np.maximum(r_maxs - r_mins, 0) / (grid[1] - grid[0])

    def project(
        self,
        rays: RayCollection,
        data: np.ndarray,
        grid: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Project ray volume data onto the vertical regions bounded by a grid.

        Parameters
        ----------
        rays
            Current ray volume properties.
        data
            The data to project (for example, pseudo-momentum fluxes). Should
            have the same shape as the properties stored in rays.
        grid, optional
            The edges of the vertical regions to project onto. If None, then the
            cell faces of the mean grid are used, such that the vertical regions
            are the cells themselves.

        Returns
        -------
        np.ndarray
            Projected values at each vertical region. Has length one shorter
            than that of the provided grid.

        """

        return np.nansum(self.get_fracs(rays, grid) * data, axis=1)

    def pmf(self, rays: RayCollection) -> np.ndarray:
        """
        Calculate the zonal and meridional pseudomomentum fluxes.

        Parameters
        ----------
        rays
            RayCollection with current ray properties.

        Returns
        -------
        np.ndarray
            Array of shape (2, config.n_grid) whose rows correspond to the zonal
            and meridional pseudomomentum fluxes, respectively, at cell faces.

        """

        cg_r = rays.cg_r()
        volume = abs(rays.dk * rays.dl * rays.dm)
        data = cg_r * rays.dens * volume

        pmf = np.zeros((2, len(self.r_faces)))
        pmf[0, 1:-1] = self.project(rays, data * rays.k, self.r_centers)
        pmf[1, 1:-1] = self.project(rays, data * rays.l, self.r_centers)

        pmf[:, 0] = pmf[:, 1]
        pmf[:, -1] = pmf[:, -2]

        if config.filter_pmf:
            pmf[:, 1:-1] = (pmf[:, :-2] + 2 * pmf[:, 1:-1] + pmf[:, 2:]) / 4

        return pmf

    def dmean_dt(self, rays: RayCollection) -> np.ndarray:
        """
        Calculate the time tendency of the mean wind, including Coriolis terms
        and pseudomomentum flux divergences from the ray volumes.

        Parameters
        ----------
        rays
            RayCollection with current ray properties.

        Returns
        -------
        np.ndarray
            Array whose first and second rows are the time tendencies of the
            zonal and meridional wind, respectively, at cell centers.
            
        """

        pmf = self.pmf(rays)
        dpmf_dr = np.diff(pmf, axis=1) / self.dr
        du_dt = config.f0 * self.v - (self.dp_dx + dpmf_dr[0]) / self.rho
        dv_dt = -config.f0 * self.u - (self.dp_dy + dpmf_dr[1]) / self.rho

        return np.vstack((du_dt, dv_dt))
