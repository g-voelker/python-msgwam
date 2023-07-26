import numpy as np
import yaml

from .constants import ROT_EARTH

boussinesq: bool
saturate_online: bool
hprop: bool

ngrid: int
grid_max: float

dt: float
nday: int
nt_max: int

phi0: float
rhobar0: float
hh: float
f0: float

N0: float
alpha: float
kappa: float

nray: int
nray_max: int

wvl_hor_init: float
wvl_ver_init: float
direction: float

dk_init: float
dl_init: float
r_m_area: float
r_init_bounds: tuple[float, float]

uv_init_method: str
u0: float
r0: float
sig_r: float

def load_config(path: str) -> None:
    """
    Load configuration settings from the provided YAML file and update the
    module namespace, so that parameters can be accessed as config.name. This
    function also does some simple precalculations of derived values that are
    used in several places.
    """

    with open(path) as f:
        config = yaml.safe_load(f)
        
    config['phi0'] = np.deg2rad(config['phi0'])
    config['f0'] = 2 * ROT_EARTH * np.sin(config['phi0'])

    config['nt_max'] = int(86400 * config['nday'] / config['dt']) + 1

    globals().update(config)
