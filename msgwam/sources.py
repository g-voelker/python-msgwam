from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from . import config

if TYPE_CHECKING:
    from .mean import MeanFlow
    from .rays import RayCollection

# Each function in this module should accept a MeanFlow and a RayCollection and
# return an array with rows corresponding to the ray properties named in
# RayCollection.props and columns corresponding to however many ray volumes
# should be launched at the source.

def desaubies(mean: MeanFlow, rays: RayCollection) -> np.ndarray:
    """
    Calculate source ray volume properties according to the Desaubies spectrum
    as defined in Boloni et al. (2021). The number of ray volumes is determined
    by n_c_tilde and n_omega_tilde, the product of which gives the number of
    rays launched in each compass direction.
    """

    ct_min, ct_max = config.c_tilde_bounds
    ot_min, ot_max = config.omega_tilde_bounds

    ct_edges = np.linspace(ct_min, ct_max, config.n_c_tilde + 1)
    ot_edges = np.linspace(ot_min, ot_max, config.n_omega_tilde + 1)

    dct = ct_edges[1] - ct_edges[0]
    dot = ot_edges[1] - ot_edges[0]

    c_tilde, omega_tilde = np.meshgrid(
        (ct_edges[:-1] + ct_edges[1:]) / 2,
        (ot_edges[:-1] + ot_edges[1:])
    )

    c_tilde = c_tilde.flatten()
    omega_tilde = omega_tilde.flatten()

    m = -config.N0 / c_tilde
    dm = m ** 2 * dct / config.N0
    wvn_hor = omega_tilde / c_tilde
    m_star = 2 * np.pi / config.wvl_ver_char

    G = m_star ** 3 * (
        (c_tilde * config.N0 ** 3 * omega_tilde ** (-2 / 3)) /
        (config.N0 ** 4 + m_star ** 4 * c_tilde ** 4)
    )

    rhobar = np.interp(config.r_launch, mean.r_centers, mean.rho)
    C = config.bc_mom_flux / (rhobar * G.sum() * dct * dot)

    n_each = config.n_c_tilde * config.n_omega_tilde
    data = np.zeros((len(rays.props), 4 * n_each))

    r = (config.r_ghost - 0.5 * config.dr_init) * np.ones(n_each)
    dr = config.dr_init * np.ones(n_each)

    for i in range(4):
        direction = i * np.pi / 2
        k = wvn_hor * np.cos(direction)
        l = wvn_hor * np.sin(direction)

        dk = -m * dot / config.N0
        dl = wvn_hor * np.pi / 2

        if i % 2 == 1:
            dk, dl = dl, dk

        cg_r = rays.cg_r(r=r, k=k, l=l, m=m)
        dens = (
            (rhobar * C * G * c_tilde ** 5) /
            (config.N0 * omega_tilde ** 2 * cg_r)
        )

        chunk = np.vstack((r, dr, k, l, m, dk, dl, dm, dens))
        data[:, (i * n_each):((i + 1) * n_each)] = chunk

    return data

def legacy(mean: MeanFlow, rays: RayCollection) -> np.ndarray:
    """
    Calculate source ray volumes as was done in the original version of this
    Python code. Note that the library defaults to not launching more rays at
    the lower boundary when this option is chosen.
    """
    
    wvn_hor = 2 * np.pi / config.wvl_hor_char
    direction = np.deg2rad(config.direction)

    k = wvn_hor * np.cos(direction) * np.ones(config.n_ray)
    l = wvn_hor * np.sin(direction) * np.ones(config.n_ray)
    m = -2 * np.pi / config.wvl_ver_char * np.ones(config.n_ray)

    r_min, r_max = config.r_init_bounds
    r_edges = np.linspace(r_min, r_max, config.n_ray + 1)
    dr = r_edges[1] - r_edges[0] * np.ones(config.n_ray)
    r = (r_edges[:-1] + r_edges[1:]) / 2

    dk = config.dk_init * np.ones(config.n_ray)
    dl = config.dl_init * np.ones(config.n_ray)
    dm = config.r_m_area / dr

    rhobar = np.interp(r, mean.r_centers, mean.rho)
    omega_hat = rays.omega_hat(k=k, l=l, m=m)

    amplitude = (
        (config.alpha ** 2 * rhobar * omega_hat * config.N0 ** 2) /
        (2 * m ** 2 * (omega_hat ** 2 - config.f0 ** 2))
    )

    profile = np.exp(-0.5 * ((r - r.mean()) / 2000) ** 2)
    dens = amplitude * profile / (dk * dl * dm)

    return np.vstack((r, dr, k, l, m, dk, dl, dm, dens))

def monochromatic(_, rays: RayCollection) -> np.ndarray:
    """
    Calculate ray properties for a single volume at prescribed wavenumber.
    """
    
    r = config.r_launch
    dr = config.dr_init
    
    wvn_hor = 2 * np.pi / config.wvl_hor_char
    direction = np.deg2rad(config.direction)

    k = wvn_hor * np.cos(direction)
    l = wvn_hor * np.sin(direction)
    m = -2 * np.pi / config.wvl_ver_char

    dk = config.dk_init
    dl = config.dl_init
    dm = config.dm_init

    volume = abs(dk * dl * dm)
    cg_r = rays.cg_r(r=r, k=k, l=l, m=m)
    dens = config.bc_mom_flux / (k * volume * cg_r)

    return np.array([r, dr, k, l, m, dk, dl, dm, dens]).reshape(-1, 1)