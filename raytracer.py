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
#     sig_r = 10000,
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
r_init_min = 0                     # initial position in vertical above ground
r_init_max = 15000                 # initial position in vertical above ground
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
    sig_r=10000,
    u0=4,
    r0=40000,
    r1=40000,
    phi0=phi0,
    kappa=1.,
    saturate_online=False
)


########################################################
# set initial condition
########################################################

# reset rays
lprop.reset_rays()

k_abs_init = 2 * np.pi / 50e3           # set absolute horizontal wave number as inverse of wavelength
direction = 90                         # set rotation of initial wave number in degrees

# set initial wave numbers
init_k = k_abs_init * np.sin(np.deg2rad(direction))
init_l = k_abs_init * np.cos(np.deg2rad(direction))
init_m = -2 * np.pi / 5e3

# set static values / arrays
r_m_area = 5e-5
init_dl = np.ones(nray) * 1e-4
init_dk = np.ones(nray) * 1e-4

lprop.set_statics(
    dl = init_dl,
    dk = init_dk,
    r_m_area = r_m_area 
)

# set grids
grid = np.linspace(0, grid_max, ngrid)  # define background model grid
grids = .5 * (grid[:-1] + grid[1:])     # staggered grid
lprop.grid = grid                       # pass grid to the propagation library
lprop.grids = grids                     # pass grid to the propagation library

# set initial background wind
init_u = lprop.velocities_sine_homogeneous(grids)
init_v = np.zeros(init_u.shape)

# set vertical grid for initial values
init_r_grid = np.linspace(r_init_min, r_init_max, nray + 1)
init_r = .5 * (init_r_grid[1:] + init_r_grid[:-1])
init_dr = np.diff(init_r_grid)[0]
init_dm = r_m_area / init_dr

# set background
lprop.set_hydrostatics()
lprop.set_pressure_gradient(init_u, init_v)

# # set wave action density
f0 = 2 * lprop.ROT_EARTH * np.sin(phi0)
rhobar_ray = np.interp(init_r, grids, lprop.rhobar)
omh_ray = lprop.omega(init_k, init_l, init_m, phi0)
amplitude = alpha**2 * rhobar_ray / 2 * omh_ray / init_m**2 / (omh_ray**2 - f0**2) * NN**2
profile = np.exp(-(init_r - init_r.mean())**2 / 2 / 2000**2)
init_dens = amplitude * profile / init_dk / init_dl / init_dm

ray_volumes = np.array([
    lprop.Ray(
        lon = 0,
        lat = phi0,
        r = init_r[nvol],
        dr = init_dr,
        k = init_k,
        l = init_l,
        m = init_m,
        dens = init_dens[nvol],
        volume = r_m_area,
    ) for nvol in range(0, nray)
])


########################################################
# initialize arrays
########################################################

# allocate all arrays
int_dens = np.zeros((nt_max + 1, nray))
int_dens_prop = np.zeros((nt_max + 1, nray))
int_lambda = np.zeros((nt_max + 1, nray))
int_phi = np.zeros((nt_max + 1, nray))
int_r = np.zeros((nt_max + 1, nray))
int_dr = np.zeros((nt_max + 1, nray))
int_k = np.zeros((nt_max + 1, nray))
int_l = np.zeros((nt_max + 1, nray))
int_m = np.zeros((nt_max + 1, nray))
int_dm = np.zeros((nt_max + 1, nray))
int_u = np.zeros((nt_max + 1, len(grids)))
int_v = np.zeros((nt_max + 1, len(grids)))

# set initial values for diagnostics
int_dens[0] = lprop.Ray.dens[:lprop.Ray.count, 0]
int_dens_prop[0] = lprop.Ray.dens[:lprop.Ray.count, 0]
int_lambda[0] = lprop.Ray.lon[:lprop.Ray.count, 0]
int_phi[0] = lprop.Ray.lat[:lprop.Ray.count, 0]
int_r[0] = lprop.Ray.r[:lprop.Ray.count, 0]
int_dr[0] = lprop.Ray.dr[:lprop.Ray.count, 0]
int_k[0] = lprop.Ray.k[:lprop.Ray.count, 0]
int_l[0] = lprop.Ray.l[:lprop.Ray.count, 0]
int_m[0] = lprop.Ray.m[:lprop.Ray.count, 0]
int_dm[0] = lprop.Ray.dm[:lprop.Ray.count, 0]
int_u[0] = init_u
int_v[0] = init_v


########################################################
# integrate model in time
########################################################

for nt in range(1, nt_max + 1):
    
    # integrate a time step
    state_out = lprop.RK3(dt, int_u[nt - 1], int_v[nt - 1])
    # print(state_out[0])

    # decompose the state vector into the data arrays for i/o
    int_dens_prop[nt], int_lambda[nt], int_phi[nt], int_r[nt], int_dr[nt], \
        int_k[nt], int_l[nt], int_m[nt], int_dm[nt], \
        int_u[nt], int_v[nt] = state_out
        
    if not lprop.model_config['saturate_online']:
        # calculate saturated wave action density
        int_dens[nt] = lprop.saturation(
            dt, int_dens_prop[nt], int_r[nt-1], (int_r[nt] - int_r[nt-1]) / dt,
            int_dr[nt-1], (int_dr[nt] - int_dr[nt-1]) / dt,
            int_k[nt], int_l[nt], int_m[nt-1],
            (int_m[nt] - int_m[nt-1]) / dt, direct=True
        )
    else:
        int_dens[nt] = int_dens_prop[nt]
        
    # output the progress in %
    print('progress: {0:.2f}%'.format(nt / nt_max * 100), end='\r')


########################################################
# diagnose wave action conservation
########################################################

nproj = [0, len(time)]

int_r_down = int_r - .5 * int_dr
int_r_up = int_r + .5 * int_dr
int_m_down = int_m - .5 * int_dm
int_m_up = int_m + .5 * int_dm

int_dk = np.ones(nray)
int_dlam = np.ones(nray)
int_dphi = np.ones(nray)


wa = np.zeros((nproj[1] - nproj[0], len(grids)))

for nt in range(nproj[0], nproj[1] - 2):
    wa[nt] = lprop.wave_projection(
        int_dens[nt], int_lambda[nt], int_phi[nt], int_r_down[nt], int_r_up[nt],
        int_k[nt], int_l[nt], int_m_down[nt], int_m_up[nt],
        init_dk, init_dl, int_dm[nt], grid, var=2
    )

wa[-1] = lprop.wave_projection(
    int_dens[nproj[1] - 1], int_lambda[nproj[1] - 1], int_phi[nproj[1] - 1], int_r_down[nproj[1] - 1],
    int_r_up[nproj[1 - 1]], int_k[nproj[1] - 1], int_l[nproj[1] - 1], int_m_down[nproj[1] - 1],
    int_m_up[nproj[1] - 1], init_dk, init_dl, int_dm[nproj[1] - 1], grid, var=2
)

flux_diag = np.zeros((nproj[1] - nproj[0] - 1, len(grids) - 1))
for nt in range(nproj[0], nproj[1] - 2):
    flux_diag[nt] = lprop.wave_projection(
        int_dens[nt], int_lambda[nt], int_phi[nt], int_r_down[nt], int_r_up[nt],
        int_k[nt], int_l[nt], int_m_down[nt], int_m_up[nt],
        init_dk, init_dl, int_dm[nt], grids, var=1
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
