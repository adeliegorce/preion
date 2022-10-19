import numpy as np
from give_RF import get_ksz
from parameters import *

# KSZ
ells_data = np.array([3000.0])
ksz_data = get_ksz(bf_values, ells_data)
ksz_err = np.array([0.1])
print("kSZ:", ells_data, ksz_data, ksz_err)
np.savetxt(
    "ksz_data.txt", np.c_[ells_data, ksz_data, ksz_err], header="ells, ksz [mK2], err"
)

# 21cm
k_data = np.array([0.50, 0.53])
z_data = np.array([7.9, 10.4])
data_21 = np.array([457, 3476])
err_21 = np.array([0, 0])
print("21cm PS:", k_data, z_data, data_21, err_21)
np.savetxt(
    "ps21_data.txt",
    np.c_[k_data, z_data, data_21, err_21],
    header="k [Mpc-1], z, 21cm ps, error bar",
)
