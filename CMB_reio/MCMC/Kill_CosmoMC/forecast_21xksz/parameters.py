h = 0.6774
T_cmb = 2.7255
omegabh2 = 0.0224
omegach2 = 0.120
thetaMC = 1.041
n_s = 0.9677
logA = 3.044
alpha0 = 3.7
kappa = 0.10
z_early = 20.0
xe_recomb = 1e-4
bf_values = [6.8, 1.5]
ref_params = [
    omegabh2,
    omegach2,
    n_s,
    thetaMC,
    logA,
    bf_values[0],
    bf_values[1],
    alpha0,
    kappa,
]

# priors equivalent to 5sigma on Planck 2018 BF values
priors = [(5.0, 10.0), (0.1, 4.0)]
zend_min = 4.5

rf_folder = "/Users/adeliegorce/Documents/ML-TKSZ/"
outroot = "test"

kmax_pk = 10.0
