import numpy as np
from main import *
import emcee
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from give_RF import get_ksz
from parameters import *
import triangleme2

matplotlib.rc("font", **{"family": "serif", "serif": ["times new roman"], "size": 16})
matplotlib.rc("axes", linewidth=1.5)
matplotlib.rc("text", usetex=True)

CL = 68
percentile1 = (100 - CL) / 2
percentile2 = CL + (100 - CL) / 2
smoothing = 1.0
props = dict(boxstyle="round", facecolor="white", alpha=0.5)

labels = [r"$z_\mathrm{re}$", r"$\mathrm{d}z$"]
paramnames = ["zre", "dz"]


########################################## DATA
print("Reading data...")
ells_data, ksz_data, ksz_err = np.loadtxt("ksz_data.txt", unpack=True)
k_data, z_data, data_21, err_21 = np.loadtxt("ps21_data.txt", unpack=True)

krange = np.unique(np.r_[np.logspace(-3, 1, 100), k_data])
lrange = np.unique(np.r_[np.linspace(100, 10000, 100), ells_data])
zrange = np.unique(np.r_[np.arange(0, 20), z_data])

########################################## CHAINS

samples = emcee.backends.HDFBackend("chains_%s.h5" % outroot, read_only=True)
ndim = samples.shape[1]
nwalkers = samples.shape[0]

chain = samples.get_chain()
like = samples.get_log_prob()
niter = chain.shape[0]  # number of iterations performed, per walker
nsamples = niter * nwalkers

print("\n%i samples for %i iterations." % (nsamples, niter))

taus = samples.get_autocorr_time(tol=0)
if np.isnan(taus).any():
    print("NaN tau. Taking 0.")
    taus = np.zeros(taus.size)
endtau = 2.0 * np.max(taus)
converged = np.all(taus * 50 < samples.iteration)
print("Auto-correlation time: %.2f. Converged: %s." % (endtau, converged))


fig, axes = plt.subplots(
    ndim + 1, 1, figsize=(6, 5), sharex=True
)  # , gridspec_kw={'hspace':0.02})
for i in range(ndim):
    ax = axes[i]
    ax.axhline(bf_values[i], color="C0", lw=1.5, ls=":")
    for j in range(nwalkers):
        axes[i].plot(chain[:, j, i], color="k", alpha=0.2)
    ax.set_ylabel(labels[i])
    ax.axvline(endtau, color="k")
for j in range(nwalkers):
    axes[-1].plot(like[:, j], color="k", alpha=0.2)
axes[-1].axvline(endtau, color="k")
axes[-1].set_ylabel(r"log$\mathcal{L}$")
axes[-1].set_xlabel("Step number")
fig.tight_layout()
fig.savefig("figures/walkers_%s.png" % outroot)

print("ML parameters:")
flatchain = samples.get_chain(flat=True, discard=int(endtau))
flatchain = np.c_[flatchain, flatchain[:, 0] - flatchain[:, 1]]
paramnames.append("zend")
labels.append(r"$z_\mathrm{end}$")
bf_values.append(bf_values[0] - bf_values[1])
for j in range(ndim):
    print(
        " %s = %.2f +/- %.2f vs %.2f"
        % (
            paramnames[j],
            np.median(flatchain[:, j]),
            np.std(flatchain[:, j]),
            bf_values[j],
        )
    )


blob_names = [
    "chi2_ksz",
    "chi2_21",
    "pksz_point",
    "p21_point",
    "model_ksz",
    "model_21",
    "tau",
]
blobs = samples.get_blobs(flat=True, discard=int(endtau))
# cmb_tt = blobs['CMB_TT']

nrand = min(flatchain.shape[0], 5000)
subsamples = np.copy(flatchain)

ksz_models = blobs["model_ksz"]
ksz_u68, ksz_l68 = np.percentile(ksz_models, percentile1, axis=0), np.percentile(
    ksz_models, percentile2, axis=0
)
ksz_med = np.median(ksz_models, axis=0)

p21_models = blobs["model_21"]

inds = np.array([0, 2], dtype=int)
fig00, axes = plt.subplots(inds.size, inds.size, figsize=(8, 6))
triangleme2.corner(
    subsamples[:, inds],
    labels=np.array(labels)[inds],
    sigmas=[1, 2],
    lw=2.0,
    plot_datapoints=False,
    color="#1f77b4",
    fig=fig00,
    smooth=2.0,
    cmap="Blues",
    truths=np.array(bf_values)[inds],
    truth_color="k",
)  # ,extents=priors)
fig00.tight_layout()
fig00.savefig("figures/triangle_plot_%s.png" % outroot)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].grid()

l, kdat, kerr = ells_data, ksz_data, ksz_err
# for l, kdat, kerr in zip(ells_data, ksz_data, ksz_err):
axes[0].errorbar(l, kdat, kerr, color="k", marker="o", capsize=3, markersize=3)
axes[0].plot(lrange, ksz_med, color="C0", lw=2.0)
axes[0].fill_between(lrange, ksz_l68, ksz_u68, color="C0", alpha=0.4)
axes[0].set_xlabel(r"Angular multipole $\ell$")
axes[0].set_ylabel(r"Patchy kSZ power $\mathcal{D}_\ell$ [$\mu$K$^2$]")


for u, (k, z, dat21, err21) in enumerate(zip(k_data, z_data, data_21, err_21)):
    axes[u + 1].grid()
    # axes[u+1].text(0.1, 100, r'$z=%.1f$' %z, bbox=props)
    uplims = False
    if err21 == 0.0:
        uplims = True
        err21 = 0.5 * dat21
    axes[u + 1].errorbar(
        k, dat21, err21, uplims=uplims, color="k", marker="o", capsize=3, markersize=3
    )
    ind1 = np.argmin(np.abs(zrange - z))
    # ind2 = np.argmin(np.abs(krange-k))

    p21_u68 = np.percentile(p21_models[:, ind1, :], percentile1, axis=0)
    p21_l68 = np.percentile(p21_models[:, ind1, :], percentile2, axis=0)
    p21_med = np.median(p21_models[:, ind1, :], axis=0)

    axes[u + 1].loglog(krange, p21_med, color="C0", lw=2.0)
    axes[u + 1].fill_between(krange, p21_l68, p21_u68, color="C0", alpha=0.4)

    axes[u + 1].set_xlabel(r"$k~[h\mathrm{Mpc}^{-1}]$")
    axes[u + 1].set_ylabel(r"$\Delta^2_{21}(k)$")
    

fig.tight_layout()
fig.savefig("figures/models_%s.png" % outroot)
