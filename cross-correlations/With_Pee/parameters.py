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
nu21_ref = 1420.4 # MHz

# REIONISATION PARAMETERS

# reionisation of Helium
helium_fullreion_redshift = 3.5
helium_fullreion_start = 5.0
helium_fullreion_deltaredshift = 0.5

# TELESCOPE SPECS
telescope_specs = {
    'SPT-3G': {'color': (0.4, 0.7607843137254902, 0.6470588235294118, 1.0)},
    'SPT-3G-Deep': {'fsky':0.036, 'fwhm':1.2, 'noise':1.6, 'lmin': 500, 'lmax':10000, 'Delta_ell':500,
               'color': (0.4, 0.7607843137254902, 0.6470588235294118, 1.0)},
    'SPT-3G-Summer': {'fsky':0.06, 'fwhm':1.2, 'noise':7.0, 'lmin': 500, 'lmax':10000, 'Delta_ell':500,
               'color': (0.4, 0.7607843137254902, 0.6470588235294118, 1.0)},
    'SPT-3G-Wide': {'fsky':0.15, 'fwhm':1.2, 'noise':9.6, 'lmin': 500, 'lmax':10000, 'Delta_ell':500,
               'color': (0.4, 0.7607843137254902, 0.6470588235294118, 1.0)},
    'AdvACT': {'fsky':0.25, 'fwhm':1.5, 'noise':13.2, 'lmin': 800, 'lmax':8500, 'Delta_ell':500,
               'color': 'C0', 'ref':'DR6'},
    'SO': {'color': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0)},
    # lmax=8000 optimistic, in forecast paper they take lmax=3000
    'SO-LAT': {'fsky':0.4, 'fwhm':1.5, 'noise':6.4, 'lmin': 1000., 'lmax': 8000.,  'Delta_ell':500,
               'ref':'https://arxiv.org/pdf/2103.02747',
               'color': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0)},  # https://arxiv.org/pdf/2103.02747
    'SO-SAT': {'fsky':0.1, 'fwhm':10.0, 'noise':2.1, 'lmax': 1000., 'lmin':10., 'Delta_ell':100,
               'ref':'https://arxiv.org/pdf/2103.02747',
               'color': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0)},  # https://arxiv.org/pdf/2412.01200
    'SO-LAT_opt': {'fsky':0.4, 'fwhm':1.5, 'noise':.6, 'lmin': 1000., 'lmax': 8000., 'Delta_ell':500, 
                   'ref':'https://arxiv.org/pdf/2103.02747',
               'color': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0)},  # https://arxiv.org/pdf/2103.02747
    'SO-SAT_opt': {'fsky':0.1, 'fwhm':10.0, 'noise':0.25, 'lmax': 1000., 'lmin':10., 'Delta_ell':100,
                   'ref':'https://arxiv.org/pdf/2103.02747',
               'color': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961, 1.0)},  # https://arxiv.org/pdf/2103.02747
    'CMB-S4': {'color': (0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.0)},
    'CMB-S4-LAT': {'fsky':0.4, 'fwhm':1.0, 'noise': 1.4142, 'lmin': 30., 'lmax': 4000., 'Delta_ell':500,
                   'ref':'https://arxiv.org/pdf/1907.04473',
                   'color': (0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.0)},
    'CMB-S4-SAT': {'fsky':0.03, 'fwhm':23., 'noise': 1., 'lmin': 5., 'lmax': 100., 'Delta_ell':10,
                   'name':'ultra-deep survey', 'ref':'https://arxiv.org/pdf/1907.04473',
                   'color': (0.5529411764705883, 0.6274509803921569, 0.796078431372549, 1.0)},
    'CMB-HD': {'fsky':0.5, 'fwhm':0.33, 'noise':.75, 'lmax':25000., 'lmin':1000., 'Delta_ell':500,
               'ref':'https://arxiv.org/pdf/1906.10134',
               'color': (0.9058823529411765, 0.5411764705882353, 0.7647058823529411, 1.0)},
    'LiteBIRD': {'fsky':.5, 'fwhm':30., 'noise':1.2, 'ref':'https://air.unimi.it/retrieve/dfa8b9a3-7df8-748b-e053-3a05fe0a3a96/BAAS_LiteBIRD_61598545769470.pdf',
                 'lmin':1., 'lmax':200., 'Delta_ell':10,
                 'color': (0.6509803921568628, 0.8470588235294118, 0.32941176470588235, 1.0)},
    'Planck': {'fsky': 0.5, 'fwhm': 8.0, 'noise': 36.0, 'lmin':30., 'lmax':2000, 'Delta_ell':50,
               'color':'k', 'ref':'PR4'},
    'PICO': {'fsky':.5, 'fwhm':7.9, 'noise':1.1, 'lmin':1., 'lmax': 4000., 'Delta_ell':100,
             'ref': 'https://arxiv.org/abs/1902.10541',
             'color': (1.0, 0.8509803921568627, 0.1843137254901961, 1.0)},
    # 'ideal': {'fsky':1., 'fwhm':.5, 'noise':.01, 'color': 'grey'}
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
