import numpy as np
import lib.libprop as lprop
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.style.use('ggplot')

########################################################
# global configuration
#
# REMINDER: The default model config is given by:
# set_model_setup(
#     u0 = 80,
#     phi0 = np.deg2rad(-60),
#     sig_phi = np.deg2rad(3),
#     rr0 = 30000,
#     rr1 = 40000,,
#     sig_rr = 10000,
#     drr = 1,
#     bvf = 0.01,
#     rhs = rhs,
#     geostrophy=True,
#     boussinesq = False,
#     hh = 8500,
#     rhobar0 = 1.2,
#     kappa = 0.95
# )
# 
########################################################

NN = 0.01                           # constant stratification for simple geometric consideration
nray = 60                           # number of rays
rr_init_min = 0                     # initial position in vertical above ground
rr_init_max = 15000                 # initial position in vertical above ground
ngrid = 101
grid_max = 100e3
lprop.HPROP_GLOBAL = False          # switch off horizontal propagation
                                    # (not implemented for nonlinear interactions)
phi0 = np.deg2rad(0)                # latitude for 1D run
alpha = 0.01                        # initial amplitude relative to static instability
plot_max = 24 * 3600                # max time for plot
plot_ymax = 100                     # max height for plot

# set time axis
dt = 120                            # time step size
nday = 2                            # number of days to integrate
nt_max = int(86400 / dt * nday)     # number of resulting time steps

time = np.linspace(0, nt_max * dt, nt_max + 1)

# adapt global model config in config dictionary
lprop.set_model_setup(
    bvf=NN,
    rhs=lprop.rhs_default,
    boussinesq=False,
    sig_rr=10000,
    u0=4,
    rr0 = 40000,
    rr1 = 40000,
    phi0 = phi0,
    kappa = 1.,
    saturate_online = False
)


########################################################
# set initial condition
########################################################

k_abs_init = 2 * np.pi / 50e3           # set absolute horizontal wave number as inverse of wavelength
direction = 90                         # set rotation of initial wave number in degrees

grid = np.linspace(0, grid_max, ngrid)  # define background model grid
grids = .5 * (grid[:-1] + grid[1:])     # staggered grid
lprop.grid = grid                       # pass grid to the propagation library
lprop.grids = grids                     # pass grid to the propagation library

# set initial values
# use initial conditions such that at each position there are two initial rays
# those have an identical zonal but inverted meridional wave number

init_kk = np.ones(nray) * k_abs_init * np.sin(np.deg2rad(direction))
init_ll = np.ones(nray) * k_abs_init * np.cos(np.deg2rad(direction))
init_mm = np.ones(nray) * -2 * np.pi / 5e3
init_lon = np.zeros(nray)
init_lat = np.ones(nray) * phi0
init_rr_grid = np.linspace(rr_init_min, rr_init_max, nray + 1)
init_rr = .5 * (init_rr_grid[:-1] + init_rr_grid[1:])
init_drr = np.ones(nray) * np.diff(init_rr)[0]
rr_mm_area = 5e-5 * init_drr
init_dmm = rr_mm_area / init_drr
init_uu = lprop.velocities_sine_homogeneous(grids)
init_vv = np.zeros(init_uu.shape)


# set background
lprop.set_hydrostatics()
lprop.set_pressure_gradient(init_uu, init_vv)

# set static arrays
init_dll = np.ones(nray) * 1e-4
init_dkk = np.ones(nray) * 1e-4

lprop.set_statics(
    dll = init_dll,
    dkk = init_dkk,
    rr_mm_area = rr_mm_area 
)

# set wave action density
f0 = 2 * lprop.ROT_EARTH * np.sin(phi0)
rhobar_ray = np.interp(init_rr, grids, lprop.rhobar)
omh_ray = lprop.omega(init_kk, init_ll, init_mm, phi0)
amplitude = alpha**2 * rhobar_ray / 2 * omh_ray / init_mm**2 / (omh_ray**2 - f0**2) * NN**2
profile = np.exp(-(init_rr - init_rr.mean())**2 / 2 / 2000**2)
init_dens = amplitude * profile / init_dkk / init_dll / init_dmm


########################################################
# initialize arrays
########################################################

# allocate all arrays
int_dens = np.zeros((nt_max + 1, nray))
int_dens_prop = np.zeros((nt_max + 1, nray))
int_lambda = np.zeros((nt_max + 1, nray))
int_phi = np.zeros((nt_max + 1, nray))
int_rr = np.zeros((nt_max + 1, nray))
int_drr = np.zeros((nt_max + 1, nray))
int_kk = np.zeros((nt_max + 1, nray))
int_ll = np.zeros((nt_max + 1, nray))
int_mm = np.zeros((nt_max + 1, nray))
int_dmm = np.zeros((nt_max + 1, nray))
int_uu = np.zeros((nt_max + 1, len(grids)))
int_vv = np.zeros((nt_max + 1, len(grids)))

# set initial values for integration
int_dens[0] = init_dens
int_dens_prop[0] = init_dens
int_lambda[0] = init_lon
int_phi[0] = init_lat
int_rr[0] = init_rr
int_drr[0] = init_drr
int_kk[0] = init_kk
int_ll[0] = init_ll
int_mm[0] = init_mm
int_dmm[0] = init_dmm
int_uu[0] = init_uu
int_vv[0] = init_vv


########################################################
# integrate model in time
########################################################

for nt in range(1, nt_max + 1):
    
    # compile the state vector for vectorized calculation
    state_in = np.array([
        int_dens[nt - 1],
        int_lambda[nt - 1],
        int_phi[nt - 1],
        int_rr[nt - 1],
        int_drr[nt - 1],
        int_kk[nt - 1],
        int_ll[nt - 1],
        int_mm[nt - 1],
        int_dmm[nt - 1],
        int_uu[nt - 1],
        int_vv[nt - 1]
    ], dtype=object)

    # integrate a time step
    state_out = lprop.RK3(dt, state_in)

    # decompose the state vector into the data arrays
    int_dens_prop[nt], int_lambda[nt], int_phi[nt], int_rr[nt], int_drr[nt], \
        int_kk[nt], int_ll[nt], int_mm[nt], int_dmm[nt], \
        int_uu[nt], int_vv[nt] = state_out
        
    if not lprop.model_config['saturate_online']:
        int_dens[nt] = lprop.saturation(
            dt, int_dens_prop[nt], int_rr[nt-1], (int_rr[nt] - int_rr[nt-1]) / 1,
            int_drr[nt-1], (int_drr[nt] - int_drr[nt-1]) / dt,
            int_kk[nt], int_ll[nt], int_mm[nt-1],
            (int_mm[nt] - int_mm[nt-1]) / dt, direct=True
        )

    # output the progress in %
    print('progress: {0:.2f}%'.format(nt / nt_max * 100), end='\r')


########################################################
# diagnose wave action conservation
########################################################

nproj = [0, len(time) - 5]

int_rr_down = int_rr - .5 * int_drr
int_rr_up = int_rr + .5 * int_drr
int_mm_down = int_mm - .5 * int_dmm
int_mm_up = int_mm + .5 * int_dmm

int_dkk = np.ones(nray)
int_dlam = np.ones(nray)
int_dphi = np.ones(nray)


wa = np.zeros((nproj[1] - nproj[0], len(grids)))

for nt in range(nproj[0], nproj[1] - 2):
    wa[nt] = lprop.wave_projection(
        int_dens[nt], int_lambda[nt], int_phi[nt], int_rr_down[nt], int_rr_up[nt],
        int_kk[nt], int_ll[nt], int_mm_down[nt], int_mm_up[nt],
        init_dkk, init_dll, int_dmm[nt], grid, var=2
    )

wa[-1] = lprop.wave_projection(
    int_dens[nproj[1] - 1], int_lambda[nproj[1] - 1], int_phi[nproj[1] - 1], int_rr_down[nproj[1] - 1],
    int_rr_up[nproj[1 - 1]], int_kk[nproj[1] - 1], int_ll[nproj[1] - 1], int_mm_down[nproj[1] - 1],
    int_mm_up[nproj[1] - 1], init_dkk, init_dll, int_dmm[nproj[1] - 1], grid, var=2
)

flux_diag = np.zeros((nproj[1] - nproj[0] - 1, len(grids) - 1))
for nt in range(nproj[0], nproj[1] - 2):
    flux_diag[nt] = lprop.wave_projection(
        int_dens[nt], int_lambda[nt], int_phi[nt], int_rr_down[nt], int_rr_up[nt],
        int_kk[nt], int_ll[nt], int_mm_down[nt], int_mm_up[nt],
        init_dkk, init_dll, int_dmm[nt], grids, var=1
    )


# calculate diagnostics
dz = np.diff(grid[:2])[0]
prop_diag = np.zeros((nproj[1] - nproj[0] - 1, len(grids)))
prop_diag[:, 1:-1] = -np.diff(flux_diag, axis=-1) / dz

# projection time
proj_time = time[nproj[0]:nproj[-1] - 1]


########################################################
# plot accuracy comparison plots
########################################################

fig_p, ax_p = plt.subplots(
    1, 2, figsize=(8, 4),
    sharex='all', sharey='all'
)

wa_scale = wa.max() * 1000
diag_scale = 1

# ax_p[0, 0].pcolormesh(proj_time / 3600, grids / 1000, int_uu[nproj[0]:nproj[-1] - 1].T,
#                    vmin=-15, vmax=15, cmap='bwr')
wa_image = ax_p[0].pcolormesh(
    proj_time / 3600,
    grids / 1000, wa[:-1].T * 1000,
    vmin=0, vmax=wa_scale
)
diag_image = ax_p[1].pcolormesh(
    proj_time / 3600, grids / 1000,
    prop_diag.T * 1000,
    vmin=-diag_scale, vmax=diag_scale, cmap='bwr'
)

ax_p[0].set_xlim(0, plot_max / 3600)
ax_p[0].set_ylim(0, plot_ymax)

ax_p[0].set_ylabel('altitude (km)')
ax_p[0].set_xlabel('time (h)')
ax_p[1].set_xlabel('time (h)')


plt.colorbar(
    wa_image, ax=ax_p[0],
    label='wave action (mJ s / m³)',
    extend='both'
)
plt.colorbar(
    diag_image, ax=ax_p[1],
    label='wave action tendency (mJ / m³)',
    extend='both'
)

fig_p.tight_layout(rect=[0, 0, 1, 1])
# fig_p.tight_layout(rect=[0, 0, .85, 1])

plt.show()
