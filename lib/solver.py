import numpy as np

from . import config
from .mean import MeanFlow
from .rays import RayCollection

class Solver:
    def __init__(self) -> None:
        """Initialize a Solver."""

        self.mean = MeanFlow()
        self.rays = RayCollection(self.mean)

        self.nt_max = int(86400 * config.nday / config.dt) + 1
        self.int_rays = np.zeros((self.nt_max, *self.rays.data.shape))
        self.int_u = np.zeros((self.nt_max, *self.mean.u.shape))
        self.int_v = np.zeros((self.nt_max, *self.mean.v.shape))

        self.int_rays[0] = self.rays.data
        self.int_u[0] = self.mean.u
        self.int_v[0] = self.mean.v

        config.doit = True

    def rhs(self, mean: MeanFlow, rays: RayCollection) -> np.ndarray:
        """
        Calculate the time derivatives of the ray properties and the zonal and
        meridional components of the mean flow.
        """

        cg_r_down = rays.cg_r(rays.r - 0.5 * rays.dr)
        cg_r_up = rays.cg_r(rays.r + 0.5 * rays.dr)
        dr_dt = (cg_r_down + cg_r_up) / 2
        ddr_dt = cg_r_up - cg_r_down

        if config.hprop:
            raise NotImplementedError('Horizontal propagation not implemented')

        else:
            dlon_dt = np.zeros(config.nray_max)
            dlat_dt = np.zeros(config.nray_max)
            dk_dt = np.zeros(config.nray_max)
            dl_dt = np.zeros(config.nray_max)

        dm_dt = rays.dm_dt(mean)
        ddm_dt = ddr_dt * rays.dm / rays.dr

        ddens_dt = np.zeros(config.nray_max)
        if config.saturate_online:
            r_next = rays.r + config.dt * dr_dt
            m_next = rays.m + config.dt * dm_dt

            dr_next = rays.dr + config.dt * ddr_dt
            dm_next = config.r_m_area / dr_next

            max_dens = rays.max_dens(mean, r_next, dr_next, m_next)
            volume = config.dk_init * config.dl_init * dm_next
            idx = rays.dens * volume > max_dens

            ddens_dt[idx] = (max_dens - rays.dens)[idx] / config.dt

        dmean_dt = mean.dmean_dt(rays)
        drays_dt = np.vstack((
            dlon_dt,
            dlat_dt,
            dr_dt,
            ddr_dt,
            dk_dt,
            dl_dt,
            dm_dt,
            ddm_dt,
            ddens_dt
        ))

        return np.array([dmean_dt, drays_dt], dtype=object)

    def step(self, state: np.ndarray) -> None:
        delta = 0
        coeffs = [
            (0, 1 / 3),
            (-5 / 9, 15 / 16),
            (-153 / 128, 8 / 15)
        ]

        for a, b in coeffs:
            delta = config.dt * self.rhs(*state) + a * delta
            state = state + b * delta

        if not config.saturate_online:
            max_dens = self.rays.max_dens(self.mean)
            volume = config.dk_init * config.dl_init * self.rays.dm
            idx = self.rays.dens * volume > max_dens

            self.rays.dens[idx] = max_dens[idx]

        return state

    def integrate(self) -> None:
        """Integrate the system for the specified number of days."""

        state = np.array([self.mean, self.rays], dtype=object)

        for i in range(1, self.nt_max):
            state = self.step(state)

            self.int_rays[i] = self.rays.data
            self.int_u[i] = self.mean.u
            self.int_v[i] = self.mean.v