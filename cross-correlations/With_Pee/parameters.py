# FIT PARAMETERS

blobnames = [
    "pksz",
    "hksz",
    "tt_cmb",
    "ee_cmb",
    "te_cmb"
    "tau",
]
# priors
z_max = 20.0

# COSMOLOGY

T_cmb = 2.7255
Yp = 0.2453
nu21_ref = 1420. # MHz

# REIONISATION PARAMETERS

# reionisation of Helium
helium_fullreion_redshift = 3.5
helium_fullreion_start = 5.0
helium_fullreion_deltaredshift = 0.5

# TELESCOPE SPECS
telescope_specs = {
    'SPT-3G': {'fsky':0.036, 'fwhm':1.3, 'noise':14.0, 'color': (0.4, 0.7607843137254902, 0.6470588235294118, 1.0)},
    'AdvACT': {'fsky':0.5, 'fwhm':1.5, 'noise':13.2, 'color': (0.4, 0.7607843137254902, 0.6470588235294118, 1.0)},
    'SO': {'fsky':0.4, 'fwhm':2.0, 'noise':3.0, 'color': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0)},
    'CMB-S4': {'fsky':0.6, 'fwhm':1.0, 'noise': 1.4142, 'color': (0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.0)},
    'CMB-HD': {'fsky':0.5, 'fwhm':0.5, 'noise':2.7, 'color': (0.9058823529411765, 0.5411764705882353, 0.7647058823529411, 1.0)},
    'LiteBIRD': {'fsky':1., 'fwhm':30., 'noise':2.4, 'color': (0.6509803921568628, 0.8470588235294118, 0.32941176470588235, 1.0)},
    'PICO': {'fsky':1., 'fwhm':7.9, 'noise':.85, 'color': (1.0, 0.8509803921568627, 0.1843137254901961, 1.0)},
    'ideal': {'fsky':1., 'fwhm':1., 'noise':.05, 'color': (0.8980392156862745, 0.7686274509803922, 0.5803921568627451, 1.0)}
}

# ANALYSIS PARAMETERS

# statistical parameters
CL = 95  # confidence interval
percentile1 = (100 - CL) / 2
percentile2 = CL + (100 - CL) / 2
smoothing = 1.0

# plotting parameters
ylabels = ['TT', 'EE', 'TE', 'pkSZ', 'hkSZ']
props = dict(boxstyle="round", facecolor="white", alpha=0.5)
colorlist = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
cmaps = ["Blues", "Oranges", "Greens", "PuRd"]
alphas = [0.5, 0.5, 0.5, 0.9]
smooth = 1.0
