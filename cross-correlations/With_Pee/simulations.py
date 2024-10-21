import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import halfnorm


def gaussian_box_from_ps(
        N, L, pk_array, ndim=2,
        phases=None,
        ):

    assert pk_array.shape[0] == 2
    pk_prior = interp1d(
        pk_array[0], pk_array[1],
        fill_value=0., bounds_error=False
    )

    k_x = (2. * np.pi) * np.fft.fftfreq(N, d=L/N)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if ndim == 1:
        kbox = np.abs(k_x)
        size = (N,)  # (2, N)
    if ndim == 2:
        kbox = np.sqrt(a)
        size = (N, N)  # (2, N, N)
    elif ndim == 3:
        kbox = np.sqrt(a[:, :, None] + np.power(k_x, 2))
        size = (N, N, N)  # (2, N, N, N)
    powerbox = np.array(pk_prior(kbox))
    dx = L/N

    # gaussian_field
    # means = np.zeros(kbox.shape)
    # widths = np.sqrt(powerbox*0.5)  # sqrt(mk2 mpc**(ndim*2))= mk mpc**ndim
    # a, b = np.random.normal(
    #     means,
    #     widths,
    #     size=size,
    # )  # Mpc3
    # u = np.fft.irfftn(
    #         (a + b * 1j),
    #         s=(kbox.shape),
    #         norm='ortho')
    means = np.zeros(kbox.shape)
    widths = np.sqrt(powerbox)
    a = halfnorm.rvs(
        loc=means,
        scale=widths/widths.max(),  # /np.sqrt(1.-2./np.pi),
        size=size,
    ) * widths.max()  # Mpc3, modules > 0
    # a = np.random.normal(
    #     means,
    #     widths,
    #     size=size,
    # ) 
    if phases is None:
        b = np.random.random(size) * 2. * np.pi  # phases
    else:
        b = np.copy(phases)
        if np.shape(b) != size:
            raise ValueError(f'Phases array should be {size}.')
    u = np.fft.irfftn(
            a * (np.cos(b) + 1j * np.sin(b)),
            s=(kbox.shape),
            norm='ortho')
    u /= dx**(ndim/2)  # Mpc**ndim
    return u.real - np.mean(u.real)


def gaussian_box_from_cl(
        N, fov, pk_array, ndim=2,
        phases=None):

    assert pk_array.shape[0] == 2
    pk_prior = interp1d(
        pk_array[0], pk_array[1],
        fill_value=0., bounds_error=False
    )

    k_x = np.pi * np.fft.fftfreq(N, d=fov/2./N)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if ndim == 1:
        kbox = np.abs(k_x)
        size = (N,)  # (2, N)
    if ndim == 2:
        kbox = np.sqrt(a)
        size = (N, N)  # (2, N, N)
    elif ndim == 3:
        kbox = np.sqrt(a[:, :, None] + np.power(k_x, 2))
        size = (N, N, N)  # (2, N, N, N)
    powerbox = np.array(pk_prior(kbox))
    dx = fov/N

    # gaussian_field
    means = np.zeros(kbox.shape)
    # widths = np.sqrt(powerbox*0.5)  # sqrt(mk2 mpc**(ndim*2))= mk mpc**ndim
    # a, b = np.random.normal(
    #     means,
    #     widths,
    #     size=size,
    # )  # Mpc3
    # u = np.fft.irfftn(
    #         (a + b * 1j),
    #         s=(kbox.shape),
    #         norm='ortho')
    widths = np.sqrt(powerbox)
    a = halfnorm.rvs(
        loc=means,
        scale=widths/widths.max(),  # /np.sqrt(1.-2./np.pi),
        size=size,
    ) * widths.max()  # Mpc3, modules
    # a = np.random.normal(
    #     means,
    #     widths,
    #     size=size,
    # )  # Mpc3, modules

    if phases is None:
        b = np.random.random(size) * 2. * np.pi  # phases
    else:
        b = np.copy(phases)
        if np.shape(b) != size:
            raise ValueError(f'Phases array should be {size}.')
    u = np.fft.irfftn(
            a * (np.cos(b) + 1j * np.sin(b)),
            s=(kbox.shape),
            norm='ortho')
    u /= dx**(ndim/2)  # Mpc**ndim

    return u.real
