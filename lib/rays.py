from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Any, Literal, Optional

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
        
        self.data = np.nan * np.zeros((len(self.props), config.n_ray_max))
        source_func = getattr(sources, config.source_method)
        self.sources: np.ndarray = source_func(mean, self)

        _, n_sources = self.sources.shape
        self.data[:, :n_sources] = self.sources
        self.ghosts = {i : i for i in range(n_sources)}

    def __add__(self, other: np.ndarray) -> RayCollection:
        
        if other.shape != self.data.shape:
            raise ValueError('other does not have correct shape')

        output = copy(self)
        output.data = self.data + other

        return output

    def __getattr__(self, prop: str) -> Any:
        
        if prop in self.indices:
            return self.data[self.indices[prop]]

        message = f'{type(self).__name__} object has no attribute {prop}'
        raise AttributeError(message)

    @property
    def valid(self) -> np.ndarray:
        return ~(np.isnan(self.data).sum(axis=0) > 0)

    @property
    def count(self) -> int:
        return self.valid.sum()

    def add_ray(
        self,
        r: float, dr: float,
        k: float, l: float, m: float,
        dk: float, dl: float, dm: float,
        dens: float
    ) -> None:

        if self.count == config.n_ray_max:
            raise RuntimeError('trying to add too many ray volumes')

        self.r[self.count] = r
        self.dr[self.count] = dr

        self.k[self.count] = k
        self.l[self.count] = l
        self.m[self.count] = m

        self.dk[self.count] = dk
        self.dl[self.count] = dl
        self.dm[self.count] = dm

        self.dens[self.count] = dens

    def omega_hat(
        self,
        k: Optional[float | np.ndarray]=None,
        l: Optional[float | np.ndarray]=None,
        m: Optional[float | np.ndarray]=None
    ) -> float | np.ndarray:

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

    def cg_r(
        self,
        r: Optional[np.ndarray]=None,
        k: Optional[np.ndarray]=None,
        l: Optional[np.ndarray]=None,
        m: Optional[np.ndarray]=None,
    ) -> np.ndarray:

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

        r = self.r if r is None else r
        m = self.m if m is None else m
        dm = self.dm if dm is None else dm

        rhobar = np.interp(r, mean.r_centers, mean.rho)
        volume = self.dk * self.dl * dm
        omega_hat = self.omega_hat(m=m)

        return (
            (0.5 * config.kappa ** 2 * rhobar * omega_hat * config.N0 ** 2) /
            (volume * m ** 2 * (omega_hat ** 2 - config.f0 ** 2))
        )

    def drays_dt(self, mean: MeanFlow) -> np.ndarray:

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
