import numpy as np
import xarray as xr

from . import config
from .mean import MeanFlow
from .rays import RayCollection

_as = [0, -5 / 9, -153 / 128]
_bs = [1 / 3, 15 / 16, 8 / 15]

def RK3(
    mean: MeanFlow,
    rays: RayCollection
) -> tuple[MeanFlow, RayCollection]:
    """Advance the state of the system with an RK3 step."""

    delta = 0
    state = np.array([mean, rays])

    for a, b in zip(_as, _bs):
        dmean_dt = mean.dmean_dt(rays)
        drays_dt = rays.drays_dt(mean)
        dstate_dt = np.array([dmean_dt, drays_dt], dtype=object)

        delta = config.dt * dstate_dt + a * delta
        state = state + b * delta

    mean, rays = state
    if not config.saturate_online:
        max_dens = rays.max_dens(mean)
        volume = abs(config.dk_init * config.dl_init * rays.dm)
        idx = rays.dens * volume > max_dens

        rays.dens[idx] = max_dens[idx]

    return mean, rays

def integrate() -> xr.Dataset:
    mean = MeanFlow()
    rays = RayCollection.from_config(mean)

    int_mean = [mean]
    int_rays = [rays]

    for _ in range(1, config.nt_max):
        mean, rays = RK3(mean, rays)
        int_mean.append(mean)
        int_rays.append(rays)

    data = {
        'time' : config.dt * np.arange(config.nt_max),
        'nray' : np.arange(config.nray_max),
        'grid' : int_mean[0].r_centers
    }
    
    for name in ['u', 'v']:
        stacked = np.vstack([getattr(mean, name) for mean in int_mean])
        data[name] = (('time', 'grid'), stacked)

    for name in RayCollection.names:
        stacked = np.vstack([getattr(rays, name) for rays in int_rays])
        data[name] = (('time', 'nray'), stacked)

    return xr.Dataset(data)