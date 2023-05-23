import numpy as np

RAD_EARTH = 6378e3          # mean Earth radius
ROT_EARTH = 7.2921e-5       # earth rotation
HPROP_GLOBAL = True         # global flag for horizontal propagation
pressure_gradient = 0       # pressure gradient to establish geostrophic balance
grid = None                 # grid for mean flow wind (only 1D at this point)
grids = None                # staggered grid for mean flow wind (only 1D at this point)
rhobar = 1                  # define hyfdrostatic pressure (only 1d background at this point)
model_config = {}           # initialize global model config
statics = {}


def set_statics(**kwargs):
    """
    Set constants for the model run. 
    Defaults are:
    
    int_dll = 1     (width in l-direction)
    int_dkk = 1     (width in k-direction)
    rr_mm_area = 0  (r-m area / phase space ray volume)
    
    """
    new_statics = locals()['kwargs']
    
    for arg in new_statics.keys():
        statics[arg] = new_statics[arg]


def set_model_setup(**kwargs):
    """
    Add all optional arguments to the model setup dictionary.
    This can be used to globally set up the parameters needed
    in the propagation routines. Existing setup properties
    (defaults) are not removed but can be changed.
    See get_model_setup() for the current configuration dictionary.
    
    usage
    libprop.model_config = set_model_setup(**kwargs)
    """
    new_setup = locals()['kwargs']
    
    for arg in new_setup.keys():
        model_config[arg] = new_setup[arg]


def set_hydrostatics():
    """
    Set the hydrostatic density given a density scale height hh.

    """
    
    global rhobar
    
    rhobar0 = model_config['rhobar0']
    hh = model_config['hh']
    boussinesq = model_config['boussinesq']
    
    if boussinesq:
        rhobar = rhobar0 * np.ones(grids.shape)
    else:
        rhobar = rhobar0 * np.exp(-grids / hh)


def set_pressure_gradient(uu, vv):
    """
    Initialize the pressure gradient based on the geostrophy option.
    The geostrophy parameter needs to be set before calling this routine.

    Args:
        uu (ndarray)    : balanced background velocity
        vv (ndarray)    : balanced background velocity
    """
    
    global pressure_gradient 
    
    phi0 = model_config['phi0']
    ff = 2 * ROT_EARTH * np.sin(phi0)
    pressure_gradient = np.empty((2, len(grids)))
    
    pressure_gradient[0] = rhobar * ff * vv
    pressure_gradient[1] = - rhobar * ff * uu


def get_model_setup():
    """
    Return the model configuration dictionary.
    """
    return model_config


def wave_projection(dens, lam, phi, rr_low, rr_up,
                    kk, ll, mm_low, mm_up, dkk, dll, dmm,
                    grid, var=0):
    """
    Project the wave property onto the grid for diagnostics and wave-mean-flow coupling. Choose one of the fololowing:
    var = 0 / pseudo-momentum fluxes at cell center
    var = 1 / vertical wave action flux at cell center
    var = 2 / wave action at cell center
    var = 3 / wave action fluxes at cell boundaries
    var = 4 / pseudo-momentum fluxes at cell boundaries

    Args:
        dens    (ndarray) : phase space wave action density of rays
        lam     (ndarray) : longitude of rays
        phi     (ndarray) : latitude of rays
        rr_low  (ndarray) : radius at lower edge of rays
        rr_up   (ndarray) : radius at upper edge of rays
        kk      (ndarray) : hor. wave number in x-direction of rays
        ll      (ndarray) : hor. wave number in y-direction of rays
        mm_low  (ndarray) : vert. wave number at lower edge of rays
        mm_up   (ndarray) : vert. wave number at upper edge of rays
        dkk     (ndarray) : ray volume extent in kk-direction
        dll     (ndarray) : ray volume extent in ll-direction
        dmm     (ndarray) : ray volume extent in mm-direction
        grid    (ndarray) : vertical grid for projection
        var     (int)     : choice of variable
    Output:
        projection (ndarray) : projected variable on grid
    """
    invalid = -99999
    
    dz = np.diff(grid[:2])[0]
    nlow = (rr_low / dz).astype(int)
    nup = (rr_up / dz + 1.).astype(int)

    nzmax = len(grid) - 2

    out_of_domain = np.where((((nlow >= nzmax) & (nup >= nzmax)) |
                              ((nlow <= 0) & (nup <= 0))))
    
    for idx in (nup, nlow):
        idx[np.where(idx < 0)] = 0
        idx[np.where(idx >= nzmax)] = nzmax
        idx[out_of_domain] = -99999

    phase_space_vol = abs(dkk * dll * dmm) #  * RAD_EARTH**2 * np.cos(phi))

    cgr = cg_rr(
        kk, ll,
        .5 * (mm_low + mm_up),
        lam, phi,
        .5 * (rr_low + rr_up)
    )

    if var == 0:
        projection = np.zeros((2, len(grid) - 1))
        var0 = cgr * kk * dens
        var1 = cgr * ll * dens

        for nr in range(0, len(dens)):
            
            if nlow[nr] == invalid:
                continue
            
            for ncell in range(nlow[nr], nup[nr]):
                zmin = np.max([grid[ncell], rr_low[nr]])
                zmax = np.min([grid[ncell + 1], rr_up[nr]])
                
                dri_o_dr = np.abs(zmax - zmin) / dz
                
                projection[0, ncell] += dri_o_dr * phase_space_vol[nr] * var0[nr]
                projection[1, ncell] += dri_o_dr * phase_space_vol[nr] * var1[nr]
                
    elif var == 1:
        projection = np.zeros((len(grid) - 1,))
        var0 = cgr * dens

        for nr in range(0, len(dens)):
            
            if nlow[nr] == invalid:
                continue
            
            for ncell in range(nlow[nr], nup[nr]):
                zmin = np.max([grid[ncell], rr_low[nr]])
                zmax = np.min([grid[ncell + 1], rr_up[nr]])
                
                dri_o_dr = np.abs(zmax - zmin) / dz
                
                projection[ncell] += dri_o_dr * phase_space_vol[nr] * var0[nr]
                
    elif var == 2:
        projection = np.zeros((len(grid) - 1,))
        var0 = dens

        for nr in range(0, len(dens)):
            
            if nlow[nr] == invalid:
                continue
            
            for ncell in range(nlow[nr], nup[nr]):
                zmin = np.max([grid[ncell], rr_low[nr]])
                zmax = np.min([grid[ncell + 1], rr_up[nr]])
                
                dri_o_dr = np.abs(zmax - zmin) / dz
                
                projection[ncell] += dri_o_dr * phase_space_vol[nr] * var0[nr]
                
    elif var == 3:  # get projection of wave action flux at interfaces
        projection = np.zeros((len(grid),))
        var0 = cgr * dens
        
        for nb in range(1, len(grid) - 1) :  # loop over boundaries given by grid
            
            index = np.where((nlow < nb) & (nup > nb) & (nlow != invalid))

            projection[nb] += np.sum(var0[index] * phase_space_vol[index])
            
    elif var == 4:  # get projection of pm flux at interfaces
        projection = np.zeros((2, len(grid)))
        var0 = cgr * kk * dens
        var1 = cgr * ll * dens
        
        for nb in range(1, len(grid) - 1) :  # loop over boundaries given by grid
            
            index = np.where((nlow < nb) & (nup > nb) & (nlow != invalid))

            projection[0, nb] += np.sum(var0[index] * phase_space_vol[index])
            projection[1, nb] += np.sum(var1[index] * phase_space_vol[index])

    return projection


def velocities_tanh(lam, phi, rr):
    """
    Return velocities for a background jet at position lambda, phi, rr.
    The shape is Gaussioan in phi and a tanh in rr.
    
    inputs
    lam, phi, rr    # position to get background velocity
    
    output
    velocity coomponents; shape(3):
    [u, v, w]
    """
    
    u0 = model_config['u0']
    phi0 = model_config['phi0']
    sig_phi = model_config['sig_phi']
    rr0 = model_config['rr0']
    sig_rr = model_config['sig_rr']

    exponential = np.exp(-(phi - phi0)**2 / 2 / sig_phi**2) * (np.tanh((rr - rr0) / sig_rr) + 1) * 0.5

    uu = u0 * exponential
    
    return_array = np.zeros((4, 3) + lam.shape)
    return_array[0] = uu
    
    return return_array


def velocities_tanh_homogeneous(rr):
    """
    Return velocities for a background jet at heights rr.
    The shape of the jet is a tanh in rr.
    
    inputs
    rr    # position to get background velocity
    
    output
    velocity u
    """
    
    u0 = model_config['u0']
    rr0 = model_config['rr0']
    sig_rr = model_config['sig_rr']

    exponential = (np.tanh((rr - rr0) / sig_rr) + 1) * 0.5

    uu = u0 * exponential

    return uu


def velocities_gauss_homogeneous(rr):
    """
    Return velocities for a background jet at heights rr.
    The shape of the jet is a tanh in rr.
    
    inputs
    rr    # position to get background velocity
    
    output
    velocity u
    """
    
    return_array = np.zeros(rr.shape)
    
    u0 = model_config['u0']
    rr0 = model_config['rr0']
    sig_rr = model_config['sig_rr']
    
    exponential = np.exp(-(rr - rr0)**2 / 2 / sig_rr**2)

    uu = u0 * exponential
    
    out_of_bounds = np.where((rr<=rr0 - 3*sig_rr) &
                             (rr>=rr0 + 3*sig_rr))
    
    uu[out_of_bounds] = 0.

    return uu


def velocities_sine_homogeneous(rr):
    """
    Return velocities for a background jet at position lambda, phi, rr.
    The shape of the jet is a tanh in rr.
    
    inputs
    rr  (ndarray): grid heights to get shape of u
    
    output
    uu  (ndarray): zonal velocity
    """
    
    u0 = model_config['u0']
    rr0 = model_config['rr0']
    sig_rr = model_config['sig_rr']

    exponential = .5 * (np.tanh((rr - rr0) / sig_rr) + 1)
    uu = u0 * exponential * np.sin(rr / sig_rr * 2 * np.pi)

    return uu


def gradients(lam_ray, phi_ray, rr_ray, uu, vv):
    """
    Calculate the of the given the horizontal velocities.

    Args:
        lam_ray (ndarray): zonal position of rays
        phi_ray (ndarray): meridional posision of rays
        rr_ray  (ndarray): vertical position of rays
        uu      (ndarray): zonal wind
        vv      (ndarray): meridional wind
        
    output
    velocity coomponents and it's gradients; shape(4, 3):
    [
        [u, v, w],
        [du/dlam, du/dphi, du/dr],
        [dv/dlam, dv/dphi, dv/dr],
        [dw/dlam, dw/dphi, dw/dr]
    ]
    """
    
    dz = np.diff(grid[:2])[0]
    
    
    du_dz = (uu[1:] - uu[:-1]) / dz
    dv_dz = (vv[1:] - vv[:-1]) / dz
    
    du_dz_ray = np.interp(rr_ray, grid[1:-1], du_dz)
    dv_dz_ray = np.interp(rr_ray, grid[1:-1], dv_dz)
    uu_ray = np.interp(rr_ray, grids, uu)
    vv_ray = np.interp(rr_ray, grids, vv)
    
    return_array = np.zeros((4, 3) + lam_ray.shape)
    return_array[0, 0] = uu_ray
    return_array[0, 1] = vv_ray
    return_array[1, 2] = du_dz_ray
    return_array[2, 2] = dv_dz_ray

    return return_array


def omega(kk, ll, mm, phi):
    """
    Calculate the intrinsic wave frequency.
    
    inputs
    kk, ll, mm      # wave vector
    phi             # latitute for Coriolis parameter
    
    output
    frequency       # intrinsic frequency
    """
    bvf = model_config['bvf']
    
    ff = 2 * ROT_EARTH * np.sin(phi)
    return np.sqrt((bvf ** 2 * (kk ** 2 + ll ** 2) + ff ** 2 * mm ** 2) / (kk ** 2 + ll ** 2 + mm ** 2))


def cg_lambda(kk, ll, mm, lam, phi, rr, uu, vv):
    """
    Calculate the zonal group vecocity.
    
    inputs
    kk, ll, mm              # wave vector
    lam, phi, rr            # position
    uu, vv                  # horizontal winds

    output
    cg_lambda               # zonal group velocity
    """
    bvf = model_config['bvf']
    
    uu_ray = np.interp(rr, grids, uu)
    
    vk_square = kk ** 2 + ll ** 2 + mm ** 2
    om = omega(kk, ll, mm, phi)
    if HPROP_GLOBAL:
        return kk / om / vk_square * (bvf ** 2 - om ** 2) + uu_ray
    else:
        return np.zeros(np.shape(kk))


def cg_phi(kk, ll, mm, lam, phi, rr, uu, vv):
    """
    Calculate the meridional group vecocity.
    
    inputs
    kk, ll, mm              # wave vector
    lam, phi, rr            # position
    uu, vv                  # horizontal winds

    output
    cg_phi                  # meridional group velocity
    """
    bvf = model_config['bvf']
    
    vv_ray = np.interp(rr, grids, vv)
    
    vk_square = kk ** 2 + ll ** 2 + mm ** 2
    om = omega(kk, ll, mm, phi)
    if HPROP_GLOBAL:
        return ll / om / vk_square * (bvf ** 2 - om ** 2) + vv_ray
    else:
        return np.zeros(np.shape(kk))


def cg_rr(kk, ll, mm, lam, phi, rr):
    """
    Calculate the radial group vecocity.
    
    inputs
    kk, ll, mm              # wave vector
    lam, phi, rr            # position

    output
    cg_r                    # radial group velocity
    """
    vk_square = kk ** 2 + ll ** 2 + mm ** 2
    ff = 2 * ROT_EARTH * np.sin(phi)
    om = omega(kk, ll, mm, phi)
    return - mm * (om ** 2 - ff ** 2) / om / vk_square


def dk_dt(kk, ll, mm, lam, phi, rr, uu, vv):
    """
    Calculate the modulation of the zonal wave number.
    
    inputs
    kk, ll, mm              # wave vector
    lam, phi, rr            # position
    uu, vv                  # horizontal winds

    output
    dk/dt                   # characteristic derivative of zonal wave number
    """
    vel = gradients(lam, phi, rr, uu, vv)
    du_dlam, dv_dlam = vel[1:3, 0]
    gradient = (kk * du_dlam + ll * dv_dlam) / (RAD_EARTH + rr) / np.cos(phi)

    if HPROP_GLOBAL:
        return kk / (RAD_EARTH + rr) * (np.tan(phi) * cg_phi(kk, ll, mm, lam, phi, rr, uu, vv)
                                        - cg_rr(kk, ll, mm, lam, phi, rr)) - gradient
    else:
        return np.zeros(np.shape(kk))


def dl_dt(kk, ll, mm, lam, phi, rr, uu, vv):
    """
    Calculate the modulation of the meridional wave number.
    
    inputs
    kk, ll, mm              # wave vector
    lam, phi, rr            # position
    uu, vv                  # horizontal winds

    output
    dl/dt                   # characteristic derivative of meridional wave number
    """
    vel = gradients(lam, phi, rr, uu, vv)
    du_dphi, dv_dphi = vel[1:3, 1]
    
    gradient = (kk * du_dphi + ll * dv_dphi) / (RAD_EARTH + rr)

    df2_dphi = 8 * ROT_EARTH**2 * np.sin(phi) * np.cos(phi) * 1

    if HPROP_GLOBAL:
        return - (ll * cg_rr(kk, ll, mm, lam, phi, rr)
                  + kk * np.tan(phi) * cg_lambda(kk, ll, mm, lam, phi, rr, uu, vv)
                  + mm**2 / 2 / omega(kk, ll, mm, phi) / (kk**2 + ll**2 + mm**2) * df2_dphi) / (RAD_EARTH + rr) \
               - gradient
    else:
        return np.zeros(np.shape(kk))


def dm_dt(kk, ll, mm, lam, phi, rr, uu, vv):
    """
    Calculate the modulation of the vertical wave number.
    
    inputs
    kk, ll, mm              # wave vector
    lam, phi, rr            # position
    uu, vv                  # horizontal winds

    output
    dm/dt                   # characteristic derivative of radial wave number
    """
    vel = gradients(lam, phi, rr, uu, vv)
    
    du_drr, dv_drr = vel[1:3, 2]
    gradient = (kk * du_drr + ll * dv_drr)

    return (kk * cg_lambda(kk, ll, mm, lam, phi, rr, uu, vv)
            + ll * cg_phi(kk, ll, mm, lam, phi, rr, uu, vv)) / (RAD_EARTH + rr) - gradient


def du_dt(vv, pm_flux_gradient):
    """
    Calculates the tendency for the zonal horizontal wind from the pseudo-momentum flux gradient and the Coriolis term.

    Args:
        vv                  (ndarray): meridional wind
        pm_flux_gradient    (ndarray): pseudo-momentum flux gradient

    Returns:
        ndarray: tendency of u
    """
    phi0 = model_config['phi0']
    ff = 2 * ROT_EARTH * np.sin(phi0)
    
    tendency = ff * vv - rhobar**-1 * (pressure_gradient[0] + pm_flux_gradient)
    
    return tendency


def dv_dt(uu, pm_flux_gradient):
    """
    Calculates the tendency for the zonal horizontal wind from the pseudo-momentum flux gradient and the Coriolis term.

    Args:
        uu                  (ndarray): zonal wind
        pm_flux_gradient    (ndarray): pseudo-momentum flux gradient

    Returns:
        ndarray: tendency of v
    """
    phi0 = model_config['phi0']
    ff = 2 * ROT_EARTH * np.sin(phi0)
    
    tendency = -ff * uu - rhobar**-1 * (pressure_gradient[1] + pm_flux_gradient)
    
    return tendency


def saturation(dt, dens, rr_center, rr_center_st, drr, drr_st, kk, ll, mm_center, mm_center_st,
               direct=False):
    """
    Calculates the cnhange of wave action density based on the wave action saturation criterium.

    Args:
        dt (float)              : time step
        dens (ndarray)          : wave action density
        rr_center (ndarray)     : ray volume height
        rr_center_st (ndarray)  : change in ray volume height
        drr (ndarray)           : ray volume height
        drr_st (ndarray)        : change of ray volume height
        kk (ndarray)            : ray volume zonal wave number
        ll (ndarray)            : ray volume meridional wave number
        mm_center (ndarray)     : ray volume radial wave number
        mm_center_st (ndarray)  : change in ray volume radial wave number

    Returns:
        (ndarray): change in wave action density to saturate within the time step
    """
    
    phi0 = model_config['phi0']
    NN = model_config['bvf']
    kappa = model_config['kappa']
    dkk = statics['dkk']
    dll = statics['dll']
    rr_mm_area = statics['rr_mm_area']
    
    ff = 2 * ROT_EARTH * np.sin(phi0)
    
    rr_final = rr_center + rr_center_st * dt
    drr_final = drr + drr_st * dt
    mm_final = mm_center + mm_center_st * dt
    dmm_final = rr_mm_area / drr_final
    rhobar_final = np.interp(rr_final, grids, rhobar)
    
    omh = omega(kk, ll, mm_center, phi0)
    
    phase_volume = dkk * dll * dmm_final
    
    max_dens_final = kappa**2 * .5 * rhobar_final * omh * NN**2 / mm_final**2 / (omh**2 - ff**2)
    
    dens_st = np.zeros(dens.shape)
    idx = np.where(max_dens_final < dens * phase_volume)
    
    if direct:
        dens_new = dens.copy()
        dens_new[idx] = max_dens_final[idx]
        
        return dens_new
    
    for nn in idx:
        dens_st[nn] = (max_dens_final[nn] - dens[nn]) / dt
    
    return dens_st


def rhs_default(dt, var_in):
    """
    Calculate the right hand side of the equation system.
    
    inputs
    var_in = dens, lam, phi, rr, drr,   # input state vector, must contain
             kk, ll, mm, dmm,           # wave action density, the position and wave vector,
             uu, vv                     # ray volume extents and the horizontal winds
    output
    right_hand_side                     # characteristic derivatives of state vector variables
    """
    dens, lam, phi, rr, drr, kk, ll, mm, dmm, uu, vv = var_in
    dkk = statics['dkk']
    dll = statics['dll']
    rr_mm_area = statics['rr_mm_area']
    saturate_online = model_config['saturate_online']
    
    cgr_up = cg_rr(kk, ll, mm, lam, phi, rr + .5 * drr)
    cgr_down = cg_rr(kk, ll, mm, lam, phi, rr - .5 * drr)

    dlam_st = cg_lambda(kk, ll, mm, lam, phi, rr, uu, vv) / (RAD_EARTH + rr) / np.cos(phi)
    dphi_st = cg_phi(kk, ll, mm, lam, phi, rr, uu, vv) / (RAD_EARTH + rr)
    drr_st = .5 * (cgr_down + cgr_up)
    ddrr_st = cgr_up - cgr_down
    dkk_st = dk_dt(kk, ll, mm, lam, phi, rr, uu, vv)
    dll_st = dl_dt(kk, ll, mm, lam, phi, rr, uu, vv)
    dmm_st = dm_dt(kk, ll, mm, lam, phi, rr, uu, vv)
    ddmm_st = dmm / drr * ddrr_st
    
    dens_st = saturate_online * saturation(
        dt, dens, rr, drr_st,
        drr, ddrr_st,
        kk, ll, mm, dmm_st
    )
    
    pm_flux = np.zeros((2, len(grid)))
    pm_flux[:, 1:-1] = wave_projection(
        dens, lam, phi, rr - .5*drr, rr + .5*drr,
        kk, ll, mm - .5*dmm, mm + .5*dmm,
        dkk, dll, dmm, grids
    )
    pm_flux[:, 0] = pm_flux[:, 1]
    pm_flux[:, -1] = pm_flux[:, -2]
    
    dz = np.diff(grid[:2])[0]
    pm_flux_gradient = (pm_flux[:, 1:] - pm_flux[:, :-1]) / dz
    
    du_st = du_dt(vv, pm_flux_gradient[0])
    dv_st = dv_dt(uu, pm_flux_gradient[1])

    right_hand_side = np.array([
        dens_st, dlam_st, dphi_st,
        drr_st, ddrr_st,
        dkk_st, dll_st,
        dmm_st, ddmm_st,
        du_st, dv_st
    ], dtype=object)

    return right_hand_side



def RK3(dt, var):
    """
    Integrate the state vector by dt with given right hand side function.
    
    inputs
    dt      (float)  : time step to integrate
    var_in  (ndarray): input state vector, must correspond to used right hand side
    
    output
    var_out (ndarray): updated state vector
    """
    rhs_ = model_config['rhs']
    
    qq = dt * rhs_(dt, var)
    var = var + qq / 3
    qq = dt * rhs_(dt, var) - 5 / 9 * qq
    var = var + 15 / 16 * qq
    qq = dt * rhs_(dt, var) - 153 / 128 * qq
    var = var + 8 / 15 * qq

    return var


# set the default setup 
set_model_setup(
    u0 = 80,
    phi0 = np.deg2rad(-60),
    sig_phi = np.deg2rad(3),
    rr0 = 30000,
    rr1 = 40000,
    sig_rr = 10000,
    drr = 1,
    bvf = 0.01,
    rhs = rhs_default,
    geostrophy=True,
    boussinesq = False,
    hh = 8500,
    rhobar0 = 1.2,
    kappa = 0.95,
    saturate_online = True
)

set_statics(
    int_dll = 1,
    int_dkk = 1,
    rr_mm_area = 0
)
