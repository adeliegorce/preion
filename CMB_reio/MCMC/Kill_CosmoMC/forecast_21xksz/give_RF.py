import pickle, sklearn
from scipy.interpolate import interp1d
import numpy as np
from parameters import *

seed = "RF-KSZ_"
ells_rf = np.loadtxt(f"{rf_folder}{seed}ells.txt")
expop, params_rf = np.loadtxt(f"{rf_folder}{seed}patchy_exponents.txt", unpack=True)
regressor_patchy = pickle.load(open(f"{rf_folder}{seed}patchy_cl_rf.pickle", "rb"))


def get_ksz(theta, ells):

    model_params = np.array(
        [omegabh2, omegach2, n_s, thetaMC, logA, theta[0], theta[1], alpha0, kappa]
    )
    # patchy
    pksz = interp1d(
        ells_rf,
        regressor_patchy.predict(model_params.reshape(1, model_params.size)).flatten(),
        kind="quadratic",
        fill_value="extrapolate",
    )(ells)
    pksz = (
        pksz * expop[0] * np.product(np.abs(model_params / params_rf[1:]) ** expop[1:])
    )
    # pksz[pksz<0] = 0.
    # late-time
    # hksz = interp1d(ells_rf,regressor_late.predict(rf_params.reshape(1, rf_params.size)).flatten(),
    #                 kind='quadratic', fill_value='extrapolate')(ells)
    # hksz = hksz*np.product(np.abs(theta)**exponents_late)

    return pksz
