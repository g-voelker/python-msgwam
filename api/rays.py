from typing import Any

import numpy as np

class RayCollection:
    def __init__(self, nray_max: int) -> None:
        self._count = 0
        self._nray_max = nray_max
        self._data = np.nan * np.zeros((len(self.names), nray_max))

    # ==========================================================================
    # methods for allowing ray properties to be accessed by name, so that we
    # can do things like
    #     rays = RayCollection(100)
    #     var = rays.dr
    # where the second line is secretly querying the appropriate row of
    # rays._data in the background, but the user gets easy access
    # ==========================================================================

    names = ['lon', 'lat', 'r', 'dr', 'dm', 'k', 'l', 'm', 'dens']
    indices = dict(zip(names, range(9)))
        
    def __getattr__(self, name: str) -> Any:
        if name in self.indices:
            return self._data[self.indices[name]]

        message = f'{type(self).__name__} object has no attribute {name}'
        raise AttributeError(message)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.indices:
            self._data[self.indices[name]] = value

        else:
            super().__setattr__(name, value)

    # ==========================================================================
    # methods for easy adding and deleting of rays to the collection
    # ==========================================================================

    def add_ray(
        self,
        lon: float, lat: float,
        r: float, dr: float, dm: float,
        k: float, l: float, m: float,
        dens: float
    ) -> None:
        """
        Fill in the appropriate columns of self._data with the given ray
        properties and increment the total ray count. Raise an error if too many
        rays are added.

        In the future, when we are deleting/merging rays, the RayCollection
        can keep track of which columns of self._data are free, and the user
        can use this function to avoid interacting with that directly.
        """

    def delete_ray(i: int) -> None:
        """
        Delete the ray at index i and mark that column as open. Decrement the
        total ray count.
        """

    # ==========================================================================
    # methods for calculating quantities for all rays at once using the
    # underlying representation in _data
    # ==========================================================================

    def omega_hat(self, N: float, f0: float) -> np.ndarray:
        """
        Calculate the intrinsic frequency of all rays in the collection given
        background stratification and Coriolis parameter. 
        """

    def cg_phi(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Calculate the zonal group velocity for each ray in the collection given
        mean flow velocities.
        """

    # likewise, methods for other group velocities, dk_dt terms, etc