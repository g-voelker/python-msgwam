import json

import numpy as np

from .rays import RayCollection

class Solver:

    def __init__(self, **cfg) -> None:
        # cfg is a dictionary which can be read from a config.json file supplied
        # by the user to control parameters

        # so we can, e.g., set up grids like
        self.grid = np.linspace(0, cfg['grid_max'], cfg['ngrid'])
        self.grids = (self.grid[:-1] + self.grid[1:]) / 2

        # set up "global" variables as instance variables, like
        if cfg['boussinesq']:
            self.rhobar = cfg['rhobar0'] * np.ones(self.grids.shape)
        else:
            self.rhobar = cfg['rhobar0'] * np.exp(-self.grids / cfg['hh'])

        # and, after reading the appropriate data from cfg and calculating the
        # necessary quantities, we can set up the ray volumes:
        lon, lat, r, dr, dm, k, l, m, dens = do_a_bunch_of_precalculations()
        self.rays = RayCollection(cfg['nray_max'])
        for i in range(cfg['nray']):
            self.rays.add_ray(lon, lat, r[i], dr, dm, k, l, m, dens[i])

        # we can also have instance fields for the current mean flow
        self.u, self.v = init_tanh_jet()

        # allocating the arrays to hold the results of the integration is simple
        # for each time step, we store an array of ray data along with u and v
        self.dt = cfg['dt']
        self.nt_max = int(cfg['nday'] * 86400 / self.dt)

        self.int_rays = np.zeros((self.nt_max + 1, *self.rays._data.shape))
        self.int_u = np.zeros((self.nt_max + 1, len(self.grids)))
        self.int_v = np.zeros((self.nt_max + 1, len(self.grids)))

        self.int_rays[0] = self.rays._data.shape
        self.int_u[0] = self.u
        self.int_v[0] = self.v

        current_step = 1

    def rhs(self) -> np.ndarray:
        """
        Calculate the right-hand side for the ray properties and u and v. Here
        we can use the various functions defined on self.rays as well as some
        other functions for PMF, etc, defined elsewhere. (Where?)
        """

    def RK3(self) -> np.ndarray:
        """
        Use self.rhs and the RK3 algorithm to return an array of the new ray
        properties and new mean flow velocities given the current state.
        """
    
    def step(self) -> None:
        """
        Take a time step.
        """

        self.rays._data, self.u, self.v = self.RK3()

        self.int_rays[self.current_step] = self.rays._data
        self.int_u[self.current_step] = self.u
        self.int_v[self.current_step] = self.v

        self.current_step += 1

# so, the usage would look something like:
if __name__ == '__main__':
    with open('config.json') as f:
        cfg = json.load(f)

    s = Solver(**cfg)
    
    for _ in range(s.nt_max):
        s.step()

    make_cool_plots_with_outputs()
    save_outputs()

    # etc etc






        