import numpy as np
import xarray as xr

from typing import Any

from . import config
from .mean import MeanFlow
from .rays import RayCollection

_as = [0, -5 / 9, -153 / 128]
_bs = [1 / 3, 15 / 16, 8 / 15]

def RK3(
    mean: MeanFlow,
    rays: RayCollection
) -> tuple[MeanFlow, RayCollection]:
    """
    Advance the state of the system with an RK3 step. Note that upper and lower
    boundary conditions are currently only enforced at the start of each step.

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

    p: float | np.ndarray = 0
    q: float | np.ndarray = 0

    rays.check_boundaries(mean)    
    for a, b in zip(_as, _bs):
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

def integrate() -> xr.Dataset:
    """
    Integrate the system for the duration specified in the config file.

    Returns
    -------
    xr.Dataset
        Dataset containing both mean flow and wave properties. The coordinates
        of the Dataset are time (shared by the mean flow and the waves), nray
        (allowing wave properties to be indexed by ray volume index), and grid
        (the centers of the cells in the mean flow grid).
    
    """
    
    mean = MeanFlow()
    rays = RayCollection(mean)

    to_centers = lambda a: np.vstack((
        np.interp(mean.r_centers, mean.r_faces, a[0]),
        np.interp(mean.r_centers, mean.r_faces, a[1]),
    ))

    int_mean = [mean]
    int_rays = [rays]
    int_pmf = [to_centers(mean.pmf(rays))]

    for _ in range(1, config.n_t_max):
        mean, rays = RK3(mean, rays)

        int_mean.append(mean)
        int_rays.append(rays)
        int_pmf.append(to_centers(mean.pmf(rays)))

    data: dict[str, Any] = {
        'time' : config.dt * np.arange(config.n_t_max),
        'nray' : np.arange(config.n_ray_max),
        'grid' : int_mean[0].r_centers
    }
    
    for name in ['u', 'v']:
        stacked = np.vstack([getattr(mean, name) for mean in int_mean])
        data[name] = (('time', 'grid'), stacked)

    for name in RayCollection.props:
        stacked = np.vstack([getattr(rays, name) for rays in int_rays])
        data[name] = (('time', 'nray'), stacked)

    stacked = np.stack(int_pmf).transpose(1, 0, 2)
    data['pmf_u'] = (('time', 'grid'), stacked[0])
    data['pmf_v'] = (('time', 'grid'), stacked[1])

    return xr.Dataset(data)