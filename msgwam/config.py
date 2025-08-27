import numpy as np
import tomllib

from .constants import ROT_EARTH

def load_config(path: str) -> None:
    """
    Load configuration data from a TOML file and update the module namespace so
    that parameters can be accessed as `config.{name}`. Also perform simple
    precalculations of values that are derived from configuration data but used
    in several places.

    Parameters
    ----------
    path
        path to TOML configuration file
        
    """

    with open(path, 'rb') as f:
        config = tomllib.load(f)
        
    config['phi0'] = np.deg2rad(config['phi0'])
    config['f0'] = 2 * ROT_EARTH * np.sin(config['phi0'])
    config['n_t_max'] = int(86400 * config['n_day'] / config['dt']) + 1

    if 'r_launch' in config:
        config['r_ghost'] = config['r_launch'] - config['dr_init']

    globals().update(config)
