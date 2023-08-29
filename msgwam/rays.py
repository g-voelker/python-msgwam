from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Any, Literal, Optional, overload

import numpy as np

from . import config, sources
from .constants import RAD_EARTH

if TYPE_CHECKING:
    from .mean import MeanFlow

class RayCollection:
    props = ['r', 'dr', 'k', 'l', 'm', 'dk', 'dl', 'dm', 'dens']
    indices = {prop: i for i, prop in enumerate(props)}

    r: np.ndarray; dr: np.ndarray
    k: np.ndarray; l: np.ndarray; m: np.ndarray
    dk: np.ndarray; dl: np.ndarray; dm: np.ndarray
    dens: np.ndarray

    def __init__(self, mean: MeanFlow) -> None:
        """
        Initialize a RayCollection based on loaded configuration settings and an
        already-initialized MeanFlow.

        Parameters
        ----------
        mean
            MeanFlow to use for calcultion of sources.

        """
        
        self.data = np.nan * np.zeros((len(self.props), config.n_ray_max))
        source_func = getattr(sources, config.source_method)
        self.sources: np.ndarray = source_func(mean, self)

        _, n_sources = self.sources.shape
        self.data[:, :n_sources] = self.sources
        self.ghosts = {i : i for i in range(n_sources)}

    def __add__(self, other: np.ndarray) -> RayCollection:
        """
        Return a new RayCollection object with the same source data but with ray
        properties equal to those of this object added to the values in other.
        Defined to make writing time-stepping routines easier.

        Parameters
        ----------
        other
            Array of data to be added to the ray properties (for example, their
            time tendencies multiplied by the time step).

        Returns
        -------
        RayCollection
            RayCollection with updated ray properties.

        Raises
        ------
        ValueError
            Indicates that other does not have the correct shape; that is, the
            shape of self.data.

        """

        if other.shape != self.data.shape:
            raise ValueError('other does not have correct shape')

        output = copy(self)
        output.data = self.data + other

        return output

    def __getattr__(self, prop: str) -> Any:
        """
        Return the row of self.data corresponding to the named ray property.
        Note that since __getattr__ is only called if an error is thrown during
        __getattribute__ (i.e. if a ray property is requested) we only have to
        handle those cases and can raise an error otherwise. This function is
        added so that ray properties can be accessed as `rays.dr`, etc.

        Parameters
        ----------
        prop
            Name of the ray property to return. Should be in self.props.

        Returns
        -------
            Corresponding row of self.data.

        Raises
        ------
        AttributeError
            Indicates that no ray property of the given name exists.

        """
        
        if prop in self.indices:
            return self.data[self.indices[prop]]

        message = f'{type(self).__name__} object has no attribute {prop}'
        raise AttributeError(message)

    @property
    def valid(self) -> np.ndarray:
        """
        Determine which columns of self.data correspond to active ray volumes.

        Returns
        -------
            Boolean array indicating whether a given column of self.data is
            tracking an active ray volume or is free to be written over.
        
        """

        return ~(np.isnan(self.data).sum(axis=0) > 0)

    @property
    def count(self) -> int:
        """
        Count the number of active ray volumes in the collection

        Returns
        -------
        int
            Number of active ray volumes propagating.

        """

        return self.valid.sum()

    def add_ray(
        self,
        r: float, dr: float,
        k: float, l: float, m: float,
        dk: float, dl: float, dm: float,
        dens: float
    ) -> int:
        """
        Add a ray to the collection, storing its data in the first available
        column of self.data. If the RayCollection already has the maximum
        allowable number of active ray volumes, delete the one with the lowest
        energy first.

        Parameters
        ----------
        r
            Vertical position of ray volume center.
        dr
            Vertical extent of ray volume.
        k, l, m
            Zonal, meridional, and vertical wavenumbers.
        dk, dl, dm
            Extent in zonal, meridional, and vertical wavenumber space.
        dens
            Wave action spectral density of ray volume.

        Returns
        -------
        int
            Index of the column of self.data where the new ray volume was added.

        """

        if self.count == config.n_ray_max:
            energy = self.omega_hat() * self.dens
            self.delete_rays(int(np.argmin(energy)))

        i = int(np.argmin(self.valid))
        self.data[:, i] = np.array([r, dr, k, l, m, dk, dl, dm, dens])

        return i

    def delete_rays(self, i: int | np.ndarray) -> None:
        """
        Mark the columns of self.data indicated by i as no longer corresponding
        to active ray volumes.

        Parameters
        ----------
        i
            Index or array of indices of ray volumes to stop tracking.

        """

        self.data[:, i] = np.nan

    def check_boundaries(self, mean: MeanFlow) -> None:
        """
        Delete rays that have traveled below the surface or above the maximum of
        the vertical grid. Then, add ray volumes as necessary to replace those
        that are now fully above the ghost level.

        Parameters
        ----------
        mean
            MeanFlow from which to draw the vertical grid.

        """

        below = self.r - 0.5 * self.dr < mean.r_faces[0]
        above = self.r + 0.5 * self.dr > mean.r_faces[-1]
        self.delete_rays(below | above)

        if config.source_method == 'legacy':
            return

        for i in list(self.ghosts.keys()):
            if self.r[i] - 0.5 * self.dr[i] > config.r_ghost:
                j = self.ghosts.pop(i)
                column = self.sources[:, j]
                column[0] = self.r[i] - 0.5 * (self.dr[i] + column[1])

                self.ghosts[self.add_ray(*column)] = j

    def omega_hat(
        self,
        k: Optional[float | np.ndarray]=None,
        l: Optional[float | np.ndarray]=None,
        m: Optional[float | np.ndarray]=None
    ) -> float | np.ndarray:
        """
        Calculate the intrinisic frequency of internal gravity waves.

        Parameters
        ----------
        k, optional
        l, optional
        m, optional
            Zonal, meridional, and vertical wavenumbers to use in the frequency
            calculation. If any of these is None, the corresponding wavenumbers
            from the rays in this collection will be used, such that calling
            `rays.omega_hat()` with no arguments returns the intrinsic frequency
            of each ray volume in the collection.

        Returns
        -------
        float or np.ndarray
            Intrinsic frequency. Only returns a float if all three of k, l, and
            m are passed in as floats (e.g. during initialization).

        """

        k = self.k if k is None else k
        l = self.l if l is None else l
        m = self.m if m is None else m

        return np.sqrt(
            (config.N0 ** 2 * (k ** 2 + l ** 2) + config.f0 ** 2 * m ** 2) /
            (k ** 2 + l ** 2 + m ** 2)
        )

    def cg_lonlat(
        self,
        coord: Literal['lon', 'lat'],
        mean: MeanFlow
    ) -> np.ndarray:
        """
        Calculate the zonal or meridional group velocity. If horizontal
        propagation is turned off in the config file, the returned group
        velocities will be zero everywhere.

        Parameters
        ----------
        coord
            Either 'lat' or 'lon', depending on whether zonal or meridional
            group velocity should be calculated.
        mean
            MeanFlow to be used during calculation.

        Returns
        -------
        np.ndarray
            Zonal or meridional group velocity of each ray in the collection.

        """

        if not config.hprop:
            return np.zeros(self.data.shape[1])

        u_or_v = {'lon' : mean.u, 'lat' : mean.v}[coord]
        k_or_l = {'lon' : self.k, 'lat' : self.l}[coord]

        wind = np.interp(self.r, mean.r_centers, u_or_v)
        wvn_sq = self.k ** 2 + self.l ** 2 + self.m ** 2
        omega_hat = self.omega_hat()

        return wind + k_or_l * (
            (config.N0 ** 2 - omega_hat ** 2) /
            (omega_hat * wvn_sq)
        )

    @overload
    def cg_r(self, r: float, k: float, l: float, m: float) -> float:
        ...

    @overload
    def cg_r(
        self,
        r: Optional[np.ndarray]=...,
        k: Optional[np.ndarray]=...,
        l: Optional[np.ndarray]=...,
        m: Optional[np.ndarray]=...,
    ) -> np.ndarray:
        ...

    def cg_r(self, r=None, k=None, l=None, m=None):
        """
        Calculate the vertical group velocity.

        Parameters
        ----------
        r, optional
            Vertical position of each wave. If None, the vertical position of
            the waves in the collection will be used.
        k, optional
        l, optional
        m, optional
            Zonal, meridional, and vertical wavenumber of each wave. If any of
            these is None, the corresponding wavenumbers of the waves in the
            collection will be used.

        Returns
        -------
        float or np.ndarray
            Vertical group velocity of each wave. Only returns a float if all of
            r, k, l, and m are passed explicitly as floats.

        """

        r = self.r if r is None else r
        k = self.k if k is None else k
        l = self.l if l is None else l
        m = self.m if m is None else m

        wvn_sq = k ** 2 + l ** 2 + m ** 2
        omega_hat = self.omega_hat(k, l, m)

        return -m * (
            (omega_hat ** 2 - config.f0 ** 2) /
            (omega_hat * wvn_sq)
        )

    def dm_dt(self, mean: MeanFlow) -> np.ndarray:
        """
        Calculate the time tendency of the vertical wavenumber.

        Parameters
        ----------
        mean
            MeanFlow from which to calculate vertical wind gradients.

        Returns
        -------
        np.ndarray
            Time tendency of the vertical wavenumber of waves in the collection.

        """

        du_dr = np.interp(self.r, mean.r_faces[1:-1], np.diff(mean.u) / mean.dr)
        dv_dr = np.interp(self.r, mean.r_faces[1:-1], np.diff(mean.v) / mean.dr)
        grad = self.k * du_dr + self.l * dv_dr

        cg_lon = self.cg_lonlat('lon', mean)
        cg_lat = self.cg_lonlat('lat', mean)

        return (self.k * cg_lon + self.l * cg_lat) / (RAD_EARTH + self.r) - grad

    def max_dens(
        self,
        mean: MeanFlow,
        r: Optional[np.ndarray]=None,
        m: Optional[np.ndarray]=None,
        dm: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Calculate the maximum allowed wave action density according to the
        saturation condition.

        Parameters
        ----------
        mean
            Current mean state of the system.
        r, optional
        m, optional
        dm, optional
            Vertical position, vertical wavenumber, and vertical wavenumber
            extent to use. If any of these are None, the respective properties
            of the waves in this collection will be used, but they can be passed
            explicitly so that future values can be used for online saturation.

        Returns
        -------
        np.ndarray
            Maximum allowed density for each ray collection.

        """

        r = self.r if r is None else r
        m = self.m if m is None else m
        dm = self.dm if dm is None else dm

        rhobar = np.interp(r, mean.r_centers, mean.rho)
        volume = abs(self.dk * self.dl * dm)
        omega_hat = self.omega_hat(m=m)

        return (
            (0.5 * config.kappa ** 2 * rhobar * omega_hat * config.N0 ** 2) /
            (volume * m ** 2 * (omega_hat ** 2 - config.f0 ** 2))
        )

    def drays_dt(self, mean: MeanFlow) -> np.ndarray:
        """
        Calculate the time tendency of each ray property.

        Parameters
        ----------
        mean
            Current mean state of the system.

        Returns
        -------
        np.ndarray
            Array of time tendencies, each row of which corresponds to the ray
            property named in self.props.

        Raises
        ------
        NotImplementedError
            Indicates that horizontal propagation was turned on in the config.
            
        """

        cg_r_down = self.cg_r(self.r - 0.5 * self.dr)
        cg_r_up = self.cg_r(self.r + 0.5 * self.dr)
        dr_dt = (cg_r_down + cg_r_up) / 2
        ddr_dt = cg_r_up - cg_r_down

        if config.hprop:
            raise NotImplementedError('horizontal propagation not implemented')

        else:
            dk_dt = np.zeros(config.n_ray_max)
            dl_dt = np.zeros(config.n_ray_max)
            ddk_dt = np.zeros(config.n_ray_max)
            ddl_dt = np.zeros(config.n_ray_max)

        dm_dt = self.dm_dt(mean)
        ddm_dt = ddr_dt * self.dm / self.dr

        ddens_dt = np.zeros(config.n_ray_max)
        if config.saturate_online:
            r_next = self.r + config.dt * dr_dt
            m_next = self.m + config.dt * dm_dt
            dm_next = self.m + config.dt * ddm_dt

            max_dens = self.max_dens(mean, r_next, m_next, dm_next)
            idx = self.dens > max_dens

            ddens_dt[idx] = (max_dens - self.dens)[idx] / config.dt

        if config.source_method != 'legacy':
            idx = self.r < config.r_launch
            dm_dt[idx] = ddr_dt[idx] = ddm_dt[idx] = 0

        return np.vstack((
            dr_dt,
            ddr_dt,
            dk_dt,
            dl_dt,
            dm_dt,
            ddk_dt,
            ddl_dt,
            ddm_dt,
            ddens_dt
        ))
