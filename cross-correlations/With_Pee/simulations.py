import numpy as np
from scipy.interpolate import interp1d

def gaussian_box_from_ps(N, L, pk_array, ndim=2):

    assert pk_array.shape[0] == 2
    pk_prior = interp1d(pk_array[0], pk_array[1], fill_value=0., bounds_error=False)

    k_x = (2. * np.pi) * np.fft.fftfreq(N, d=L/N)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if ndim == 1:
        kbox = np.abs(k_x)
        size = (2, N)
    if ndim == 2:
        kbox = np.sqrt(a)
        size = (2, N, N)
    elif ndim == 3:
        kbox = np.sqrt(a[:, :, None] + np.power(k_x, 2))
        size = (2, N, N, N)
    powerbox = np.array(pk_prior(kbox))
    dx = L/N

    # gaussian_field
    means = np.zeros(kbox.shape)
    widths = np.sqrt(powerbox*0.5)  # sqrt(mk2 mpc**(ndim*2))= mk mpc**ndim
    a, b = np.random.normal(
        means,
        widths,
        size=size,
    )  # Mpc3
    u = np.fft.irfftn(
            (a + b * 1j),
            s=(kbox.shape),
            norm='ortho')
    u /= dx**(ndim/2)  # Mpc**ndim

    return u.real - np.mean(u.real)


def gaussian_box_from_cl(N, fov, pk_array, ndim=2):

    assert pk_array.shape[0] == 2
    pk_prior = interp1d(pk_array[0], pk_array[1], fill_value=0., bounds_error=False)

    k_x = np.pi * np.fft.fftfreq(N, d=fov/N)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if ndim == 1:
        kbox = np.abs(k_x)
        size = (2, N)
    if ndim == 2:
        kbox = np.sqrt(a)
        size = (2, N, N)
    elif ndim == 3:
        kbox = np.sqrt(a[:, :, None] + np.power(k_x, 2))
        size = (2, N, N, N)
    powerbox = np.array(pk_prior(kbox))
    dx = fov/N

    # gaussian_field
    means = np.zeros(kbox.shape)
    widths = np.sqrt(powerbox*0.5)  # sqrt(mk2 mpc**(ndim*2))= mk mpc**ndim
    a, b = np.random.normal(
        means,
        widths,
        size=size,
    )  # Mpc3
    u = np.fft.irfftn(
            (a + b * 1j),
            s=(kbox.shape),
            norm='ortho')
    u /= dx**(ndim/2)  # Mpc**ndim

    return u.real