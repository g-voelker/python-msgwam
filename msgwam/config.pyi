def load_config(path: str) -> None:
    ...

boussinesq: bool
saturate_online: bool
filter_pmf: bool
hprop: bool

dt: float
n_day: int
n_t_max: int

n_grid: int
grid_max: float

phi0: float
rhobar0: float
hh: float
f0: float

N0: float
alpha: float
kappa: float
nu: float

uv_init_method: str
u0: float
r0: float
sig_r: float

source_method: str
n_ray_max: int
n_ray: int

wvl_hor_char: float
wvl_ver_char: float
direction: float

dk_init: float
dl_init: float
dm_init: float
r_m_area: float

r_init_bounds: tuple[float, float]
c_tilde_bounds: tuple[float, float]
omega_tilde_bounds: tuple[float, float]

n_c_tilde: int
n_omega_tilde: int
bc_mom_flux: float

r_launch: float
dr_init: float
r_ghost: float