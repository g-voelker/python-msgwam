from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from . import config
from .constants import RAD_EARTH

if TYPE_CHECKING:
    from .mean import MeanFlow

class RayCollection:
    names = ['lon', 'lat', 'r', 'dr', 'k', 'l', 'm', 'dm', 'dens']
    indices = dict(zip(names, range(len(names))))

    lon: np.ndarray
    lat: np.ndarray

    r: np.ndarray
    dr: np.ndarray
    dm: np.ndarray

    k: np.ndarray
    l: np.ndarray
    m: np.ndarray

    dens: np.ndarray

    def __init__(self, mean: MeanFlow) -> None:
        """
        Initialize a RayCollection. The density profile in the MeanFlow object
        passed as an argument will be used to initialize rays according to the
        configuration settings.
        """

        self.count = 0

        n_properties = len(self.names)
        self.data = np.nan * np.zeros((n_properties, config.nray_max))

        self.init_rays(mean)

    def __getattr__(self, name: str) -> Any:
        """
        Return the row of self.data corresponding to the given name. If there is
        no such row, raise the appropriate AttributeError. Note that __getattr__
        is called only if the usual __getattribute__ throws an error, so we only
        have to handle ray property retrieval here.
        """

        if name in self.indices:
            return self.data[self.indices[name]]

        message = f'{type(self).__name__} object has no attribute {name}'
        raise AttributeError(message)

    def __add__(self, other: np.ndarray) -> RayCollection:
        """
        Add the data in an array to the data underlying this collection and
        return the collection. Defined to make time-stepping easier.
        """

        if other.shape != self.data.shape:
            raise ValueError('other does not have correct shape')

        self.data += other

        return self

    def init_rays(self, mean: MeanFlow):
        """Initialize ray volumes according to the configuration settings."""

        k_abs = 2 * np.pi / config.wvl_hor_init
        direction = np.deg2rad(config.direction)

        k = k_abs * np.sin(direction)
        l = k_abs * np.cos(direction)
        m = -2 * np.pi / config.wvl_ver_init

        r_min, r_max = config.r_init_bounds
        r_edges = np.linspace(r_min, r_max, config.nray + 1)
        r = (r_edges[:-1] + r_edges[1:]) / 2

        dr = r_edges[1] - r_edges[0]
        dm = config.r_m_area / dr

        rhobar = np.interp(r, mean.r_centers, mean.rho)
        omega_hat = self.omega_hat(k, l, m)

        amplitude = (
            (config.alpha ** 2 * rhobar * omega_hat * config.N0 ** 2) /
            (2 * m ** 2 * (omega_hat **2 - config.f0 ** 2))
        )

        profile = np.exp(-0.5 * ((r - r.mean()) / 2000) ** 2)
        dens = amplitude * profile / (config.dk_init * config.dl_init * dm)

        for i in range(config.nray):
            self.add_ray(
                lon=0, lat=config.phi0,
                r=r[i], dr=dr,
                k=k, l=l, m=m,
                dens=dens[i]
            )

    def add_ray(
        self,
        lon: float, lat: float,
        r: float, dr: float,
        k: float, l: float, m: float,
        dens: float
    ) -> None:
        """
        Add a ray to the collection. dm will be chosen in accordance with the
        value provided for config.r_m_area. 
        """

        self.lon[self.count] = lon
        self.lat[self.count] = lat

        self.r[self.count] = r
        self.dr[self.count] = dr
        self.dm[self.count] = config.r_m_area / dr

        self.k[self.count] = k
        self.l[self.count] = l
        self.m[self.count] = m

        self.dens[self.count] = dens

        self.count += 1

    def omega_hat(
        self,
        k: Optional[float | np.ndarray]=None,
        l: Optional[float | np.ndarray]=None,
        m: Optional[float | np.ndarray]=None
    ) -> float | np.ndarray:
        """
        Calculate waves' intrinsic frequencies. If any components of the wave
        vector are not passed as arguments, the wavenumbers associated with the
        rays in the collection will be used. (Passing wavenumbers as arguments
        is allowed mainly so that this code can be used during initialization.)
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
        Calculate horizontal group velocities for each ray in either the zonal
        or meridional direction, as specified by the coord argument.
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
        
    def cg_r(self, r: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Calculate radial velocities for each ray in the collection. A radial
        position for each ray can optionally be specified, to allow calculations
        at the interfaces. Otherwise, the positions in self.r will be used.
        """

        if r is None:
            r = self.r

        wvn_sq = self.k ** 2 + self.l ** 2 + self.m ** 2
        omega_hat = self.omega_hat()

        return -self.m * (
            (omega_hat ** 2 - config.f0 ** 2) /
            (omega_hat * wvn_sq)
        )

    def dm_dt(self, mean: MeanFlow) -> np.ndarray:
        """Calculate the time derivative of the vertical wavenumber."""

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
        dr: Optional[np.ndarray]=None,
        m: Optional[np.ndarray]=None
    ) -> np.ndarray:
        """
        Calculate the maximum wave action density allowed for each wave based on
        the wave action saturation criterion. r, dr, and m can be passed in as
        optional arguments, so that future values can be used for online
        saturation. Otherwise, the values from the collection will be used.
        """

        r = self.r if r is None else r
        dr = self.dr if dr is None else r
        m = self.m if m is None else m

        rhobar = np.interp(r, mean.r_centers, mean.rho)
        omega_hat = self.omega_hat(m=m)

        return (
            (0.5 * config.kappa ** 2 * rhobar * omega_hat * config.N0 ** 2) /
            (m ** 2 * (omega_hat ** 2 - config.f0 ** 2))
        )
