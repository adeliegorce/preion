import matplotlib.pyplot as plt
from matplotlib import rc, colormaps, colors
import numpy as np


def plot_field(box, L, iz=None, label='', **kwargs):

    N = box.shape[0]
    if box.ndim >= 3:
        if iz is None:
            iz = N//2
        image = box[..., iz]
    else:
        image = np.copy(box)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        image,
        extent=(0, L, 0, L),
        origin='lower',
        **kwargs
        )
    cbar = plt.colorbar(im, ax=ax, fraction=0.05)
    if len(label) > 0:
        cbar.set_label(label)
    # ax.set_yticks(ax.get_xticks())
    ax.set_ylabel(r'$y$ [Mpc]')
    ax.set_xlabel(r'$x$ [Mpc]')
    return fig, ax


def plot_field_theta(box, theta, iz=None, label='', **kwargs):

    N = box.shape[0]
    if box.ndim > 2:
        if iz is None:
            iz = N//2
        image = box[..., iz]
    else:
        image = np.copy(box)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        image,
        extent=(-theta/2, theta/2, -theta/2, theta/2),
        origin='lower',
        **kwargs
        )
    cbar = plt.colorbar(im, ax=ax, fraction=0.05)
    if len(label) > 0:
        cbar.set_label(label)
    # ax.set_yticks(ax.get_xticks())
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_xlabel(r'$\theta$ [deg]')
    return fig, ax


def compute_ps1d(field, L, nbins=30):

    field = np.copy(field)
    N = float(np.shape(field)[0])

    deltax = L/N
    maxk = 0.5 * 2.*np.pi/deltax  # Nyquist-Shannon

    # k-grid, units 1/Mpc
    k_x = np.fft.fftfreq(int(N), d=deltax/2./np.pi)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if field.ndim == 2:
        k_norm = np.sqrt(a)
    else:
        k_norm = np.sqrt(a[:, :, None] + np.power(k_x, 2))

    # Fourier transform
    delta_k = np.fft.fftn(field, norm='ortho') * deltax**(field.ndim/2)
    # normalisation is 1/sqrt(N)
    # unit Mpc**(ndim/2)
    # power spectrum
    PS = np.real(delta_k*np.conjugate(delta_k))  # unit Mpc**ndim

    kbins = np.linspace(4*np.pi/L, maxk, nbins)
    dk = np.diff(kbins).mean()
    kbin_edges = np.r_[kbins - dk / 2, kbins.max()+dk/2]

    count = np.histogram(k_norm, kbin_edges)[0]
    count = np.array(count, dtype=float)
    k = np.histogram(
        k_norm,
        kbin_edges,
        weights=k_norm
        )[0]
    # k = np.where(count != 0, k/count, 0.)
    k = np.multiply(k, np.where(count != 0, 1./count, 0.))
    pk = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS
        )[0]
    # pk = np.where(count != 0, pk/count, 0.)
    pk = np.multiply(pk, np.where(count != 0, 1/count, 0.))
    pk2 = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS**2
        )[0]
    # pk2 = np.where(count != 0, pk2/count, 0.)
    pk2 = np.multiply(pk2, np.where(count != 0, 1./count, 0.))
    # poissonian rms
    # pk_err = np.where(count != 0, np.sqrt(pk2-pk**2)/np.sqrt(count), 0.)
    pk_err = np.multiply(
        np.sqrt(pk2-pk**2),
        np.where(count != 0, 1./np.sqrt(count), 0.))

    return k, pk, pk_err


def compute_cl_from_box(field, theta, nbins=30):

    field = np.copy(field)
    N = float(np.shape(field)[0])

    deltax = theta/N
    maxk = 0.5 * 2.*np.pi/deltax  # Nyquist-Shannon

    # k-grid, units 1/Mpc
    k_x = np.fft.fftfreq(int(N), d=deltax/np.pi/2.)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if field.ndim == 2:
        k_norm = np.sqrt(a)
    else:
        k_norm = np.sqrt(a[:, :, None] + np.power(k_x, 2))

    # Fourier transform
    delta_k = np.fft.fftn(field, norm='ortho') * deltax**(field.ndim/2)
    # normalisation is 1/sqrt(N)
    # unit Mpc**(ndim/2)
    # power spectrum
    PS = np.real(delta_k*np.conjugate(delta_k))  # unit Mpc**ndim

    kbins = np.linspace(4*np.pi/theta, maxk, nbins)
    dk = np.diff(kbins).mean()
    kbin_edges = np.r_[kbins - dk / 2, kbins.max()+dk/2]

    count = np.histogram(k_norm, kbin_edges)[0]
    count = np.array(count, dtype=float)
    k = np.histogram(
        k_norm,
        kbin_edges,
        weights=k_norm
        )[0]
    # k = np.where(count != 0, k/count, 0.)
    k = np.multiply(k, np.where(count > 0., 1./count, 0.))
    pk = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS
        )[0]
    # pk = np.where(count != 0, pk/count, 0.)
    pk = np.multiply(pk, np.where(count > 0, 1/count, 0.))
    pk2 = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS**2
        )[0]
    # pk2 = np.where(count != 0, pk2/count, 0.)
    pk2 = np.multiply(pk2, np.where(count > 0, 1./count, 0.))
    # poissonian rms
    # pk_err = np.where(count != 0, np.sqrt(pk2-pk**2)/np.sqrt(count), 0.)
    pk_err = np.multiply(
        np.sqrt(pk2-pk**2),
        np.where(count != 0, 1./np.sqrt(count), 0.))

    return k, pk, pk_err


def compute_cross_spectrum(field1, field2, L, nbins=30):

    field1 = np.copy(field1)
    field2 = np.copy(field2)
    N = float(np.shape(field2)[0])
    assert field2.ndim == field1.ndim
    assert field1.shape == field2.shape

    deltax = L/N
    maxk = 0.5 * 2.*np.pi/deltax  # Nyquist-Shannon

    # k-grid, units 1/Mpc
    k_x = np.fft.fftfreq(int(N), d=deltax/2./np.pi)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if field2.ndim == 2:
        k_norm = np.sqrt(a)
    else:
        k_norm = np.sqrt(a[:, :, None] + np.power(k_x, 2))

    # Fourier transform
    delta_k1 = np.fft.fftn(field1, norm='ortho') * deltax**(field1.ndim/2)
    delta_k2 = np.fft.fftn(field2, norm='ortho') * deltax**(field2.ndim/2)
    # normalisation is 1/sqrt(N)
    # unit Mpc**(ndim/2)
    # power spectrum
    PS = np.real(delta_k1*np.conjugate(delta_k2))  # unit Mpc**ndim

    kbins = np.linspace(4*np.pi/L, maxk, nbins)
    dk = np.diff(kbins).mean()
    kbin_edges = np.r_[kbins - dk / 2, kbins.max()+dk/2]

    count = np.histogram(k_norm, kbin_edges)[0]
    count = np.array(count, dtype=float)
    k = np.histogram(
        k_norm,
        kbin_edges,
        weights=k_norm
        )[0]
    # k = np.where(count != 0, k/count, 0.)
    k = np.multiply(k, np.where(count != 0, 1./count, 0.))
    pk = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS
        )[0]
    # pk = np.where(count != 0, pk/count, 0.)
    pk = np.multiply(pk, np.where(count != 0, 1/count, 0.))
    pk2 = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS**2
        )[0]
    # pk2 = np.where(count != 0, pk2/count, 0.)
    pk2 = np.multiply(pk2, np.where(count != 0, 1./count, 0.))
    # poissonian rms
    # pk_err = np.where(count != 0, np.sqrt(pk2-pk**2)/np.sqrt(count), 0.)
    pk_err = np.multiply(
        np.sqrt(pk2-pk**2),
        np.where(count != 0, 1./np.sqrt(count), 0.))

    return k, pk, pk_err


def compute_cross_angular_spectrum(field1, field2, theta, nbins=30):

    field1 = np.copy(field1)
    field2 = np.copy(field2)
    N = float(np.shape(field2)[0])
    assert field2.ndim == field1.ndim
    assert field1.shape == field2.shape

    deltax = theta/N
    maxk = 0.5 * 2.*np.pi/deltax  # Nyquist-Shannon

    # k-grid, units 1/Mpc
    k_x = np.fft.fftfreq(int(N), d=deltax/2./np.pi)
    a = np.power(k_x, 2)[:, None] + np.power(k_x, 2)
    if field2.ndim == 2:
        k_norm = np.sqrt(a)
    else:
        k_norm = np.sqrt(a[:, :, None] + np.power(k_x, 2))

    # Fourier transform
    delta_k1 = np.fft.fftn(field1, norm='ortho') * deltax**(field1.ndim/2)
    delta_k2 = np.fft.fftn(field2, norm='ortho') * deltax**(field2.ndim/2)
    # normalisation is 1/sqrt(N)
    # unit Mpc**(ndim/2)
    # power spectrum
    PS = np.real(delta_k1*np.conjugate(delta_k2))  # unit Mpc**ndim

    kbins = np.linspace(4*np.pi/theta, maxk, nbins)
    dk = np.diff(kbins).mean()
    kbin_edges = np.r_[kbins - dk / 2, kbins.max()+dk/2]

    count = np.histogram(k_norm, kbin_edges)[0]
    count = np.array(count, dtype=float)
    k = np.histogram(
        k_norm,
        kbin_edges,
        weights=k_norm
        )[0]
    # k = np.where(count != 0, k/count, 0.)
    k = np.multiply(k, np.where(count > 0., 1./count, 0.))
    pk = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS
        )[0]
    # pk = np.where(count != 0, pk/count, 0.)
    pk = np.multiply(pk, np.where(count > 0, 1/count, 0.))
    pk2 = np.histogram(
        k_norm,
        kbin_edges,
        weights=PS**2
        )[0]
    # pk2 = np.where(count != 0, pk2/count, 0.)
    pk2 = np.multiply(pk2, np.where(count > 0, 1./count, 0.))
    # poissonian rms
    # pk_err = np.where(count != 0, np.sqrt(pk2-pk**2)/np.sqrt(count), 0.)
    pk_err = np.multiply(
        np.sqrt(pk2-pk**2),
        np.where(count > 0, 1./np.sqrt(count), 0.))

    return k, pk, pk_err


def bin_spectrum(ls, spectrum, bin_edges):
    binned_hist = np.zeros(bin_edges.size-1)
    for ib in range(bin_edges.size-1):
        m = (ls >= bin_edges[ib]) & (ls < bin_edges[ib+1])
        if m.any():
            binned_hist[ib] = np.mean(spectrum[m])
    return binned_hist
