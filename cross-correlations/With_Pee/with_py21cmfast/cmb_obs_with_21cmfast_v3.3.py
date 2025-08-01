import py21cmfast as p21c
from astropy import cosmology, constants

print(f"21cmFAST version is {p21c.__version__}")

folder = '/data/cluster/agorce/21cmFAST_cache/'
p21c.config['direc'] = folder

# Cosmology
cos = cosmology.Planck18
Yp = 0.2453 # primordial fraction of He
Xp = 1. - Yp # primordial fraction of H
mh = constants.m_n #kg, mass of H atom
rhoc = cos.critical_density0.si #kg m-3, critical density of the Universe
nh = Xp*cos.Ob0*rhoc/mh  # m-3, number density of H 
fH = 1.08 #ignore He reion at this point

T_cmb = 2.7255
T_CMB_uK=T_cmb*1e6  


# %%
zmin = 4.9
zmax = 15.0

# N = 1070
# L = 1500.
N = 512//2
L = 780./2.

user_params = p21c.UserParams(
    HII_DIM=N, BOX_LEN=L, KEEP_3D_VELOCITIES=True, N_THREADS=5, USE_INTERPOLATION_TABLES=True,
)

lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
    min_redshift=zmin,
    max_redshift=zmax,
    quantities=('brightness_temp', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'xH_box', ),
    resolution=user_params.cell_size,
    # index_offset=0,
)

lc = p21c.run_lightcone(
    lightconer=lcn,
    global_quantities=("brightness_temp", 'density', 'xH_box'),
    direc=folder,
    user_params=user_params,
    astro_params=p21c.AstroParams({"ALPHA_STAR":0.614, "F_STAR10":-1.42, "F_ESC10":-1.78, "ALPHA_ESC":0.474, "M_TURN":8.62, "t_STAR":0.392}),
    cosmo_params=p21c.CosmoParams(hlittle=0.67, OMm=0.321, OMb=0.049, SIGMA_8=0.81, POWER_INDEX=0.963),
    flag_options={"USE_MASS_DEPENDENT_ZETA":True}
    random_seed=1,
)
 lc.save('/home/agorce/scratch/big_lightcone_21cmfast_Ivan.h5')

#lc_ref = p21c.LightCone.read('/home/agorce/scratch/big_lightcone_21cmfast.h5')
#lc = p21c.run_lightcone(
#    lightconer=lcn,
#    global_quantities=("brightness_temp", 'density', 'xH_box'),
#    direc=folder,
#    user_params=lc_ref.user_params,
#    astro_params=lc_ref.astro_params,
#    cosmo_params=lc_ref.cosmo_params,
#    flag_options=lc_ref.flag_options,
#    random_seed=lc_ref.random_seed,
#)
#lc.save('/home/agorce/scratch/big_lightcone_21cmfast_2.h5')

