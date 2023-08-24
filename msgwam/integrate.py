from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
from typing import Any

import numpy as np
import xarray as xr

from scipy.linalg import lu_factor, lu_solve

from . import config
from .mean import MeanFlow
from .rays import RayCollection

class Integrator(ABC):
    def __init__(self) -> None:
        """
        Initialize the Integrator and the arrays that will hold snapshots of the
        mean flow and the waves when the system is integrated.
        """

        mean = MeanFlow()
        rays = RayCollection(mean)
        
        self.int_mean = [mean]
        self.int_rays = [rays]
        self.int_pmf = [self.center(mean.pmf(rays))]
        
    @abstractmethod
    def step(
        self,
        mean: MeanFlow,
        rays: RayCollection
    ) -> tuple[MeanFlow, RayCollection]:
        """
        Advance the state of the system by one time step. Should be implemented
        by every Integrator subclass.

        Parameters
        ----------
        mean
            Current MeanFlow.
        rays
            Current RayCollection.

        Returns
        -------
        MeanFlow
            Updated MeanFlow.
        RayCollection
            Updated RayCollection.

        """

        pass

    def integrate(self) -> Integrator:
        """
        Integrate the system over the time interval specified in config.

        Returns
        -------
        Integrator
            The integrated system.

        """

        mean = self.int_mean[0]
        rays = self.int_rays[0]

        for _ in range(1, config.n_t_max):
            rays.check_boundaries(mean)
            mean, rays = self.step(mean, rays)

            self.int_mean.append(mean)
            self.int_rays.append(rays)
            self.int_pmf.append(self.center(mean.pmf(rays)))

        return self

    def center(self, pmf: np.ndarray) -> np.ndarray:
        r_from = self.int_mean[0].r_faces
        r_to = self.int_mean[0].r_centers

        return np.vstack((
            np.interp(r_to, r_from, pmf[0]),
            np.interp(r_to, r_from, pmf[1])
        ))

    def to_netcdf(self, output_path: str) -> None:
        """
        Save a netCDF file holding the data from the integrated system.

        Parameters
        -----------
        output_path
            Where to save the netCDF file.

        """

        data: dict[str, Any] = {
            'time' : config.dt * np.arange(config.n_t_max),
            'nray' : np.arange(config.n_ray_max),
            'grid' : self.int_mean[0].r_centers
        }
        
        for name in ['u', 'v']:
            stacked = np.vstack([getattr(mean, name) for mean in self.int_mean])
            data[name] = (('time', 'grid'), stacked)

        for name in RayCollection.props:
            stacked = np.vstack([getattr(rays, name) for rays in self.int_rays])
            data[name] = (('time', 'nray'), stacked)

        stacked = np.stack(self.int_pmf).transpose(1, 0, 2)
        data['pmf_u'] = (('time', 'grid'), stacked[0])
        data['pmf_v'] = (('time', 'grid'), stacked[1])

        xr.Dataset(data).to_netcdf(output_path)
    
class RK3Integrator(Integrator):
    aa = [0, -5 / 9, -153 / 128]
    bb = [1 / 3, 15 / 16, 8 / 15]

    def __init__(self) -> None:
        super().__init__()

    def step(
        self,
        mean: MeanFlow,
        rays: RayCollection
    ) -> tuple[MeanFlow, RayCollection]:
        """
        Take an RK3 step.
        """

        p: float | np.ndarray = 0
        q: float | np.ndarray = 0

        for a, b in zip(self.aa, self.bb):
            dmean_dt = mean.dmean_dt(rays)
            drays_dt = rays.drays_dt(mean)

            p = config.dt * dmean_dt + a * p
            q = config.dt * drays_dt + a * q

            mean = mean + b * p
            rays = rays + b * q

        if not config.saturate_online:
            max_dens = rays.max_dens(mean)
            idx = rays.dens > max_dens
            rays.dens[idx] = max_dens[idx]

        return mean, rays