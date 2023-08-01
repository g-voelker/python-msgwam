from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from . import config

if TYPE_CHECKING:
    from .mean import MeanFlow
    from .rays import RayCollection

def desaubies(mean: MeanFlow, rays: RayCollection) -> np.ndarray:
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

    r_ghost = config.r_launch - config.dr_init
    r = (r_ghost - 0.5 * config.dr_init) * np.ones(n_each)
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

def simple(mean: MeanFlow, rays: RayCollection) -> np.ndarray:
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
