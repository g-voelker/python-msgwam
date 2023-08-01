import numpy as np
import tomllib

from .constants import ROT_EARTH

def load_config(path: str) -> None:
    """
    Load configuration settings from the provided YAML file and update the
    module namespace, so that parameters can be accessed as config.name. This
    function also does some simple precalculations of derived values that are
    used in several places.
    """

    with open(path, 'rb') as f:
        config = tomllib.load(f)
        
    config['phi0'] = np.deg2rad(config['phi0'])
    config['f0'] = 2 * ROT_EARTH * np.sin(config['phi0'])

    config['n_t_max'] = int(86400 * config['n_day'] / config['dt']) + 1

    globals().update(config)
