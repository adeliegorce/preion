import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import warnings
from matplotlib import cm, colors
import camb
from astropy import cosmology
from scipy.special import spherical_jn


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def get_tree(box):

    ndim = box.ndim
    N = box.shape[0]
    assert np.unique(box.shape).size == 1
    if ndim == 2:
        # slice_im = box
        x, y = np.mgrid[0:N, 0:N]
        points = np.c_[x.ravel(), y.ravel()]
    elif ndim == 3:
        # slice_im = box[..., N//2]
        x, y, z = np.mgrid[0:N, 0:N, 0:N]
        points = np.c_[x.ravel(), y.ravel(), z.ravel()]
    tree = KDTree(points)
    return tree


def get_count_in_cells(box, radius_px, tree=None):

    if tree is None:
        tree = get_tree(box)
    else:
        assert isinstance(tree, KDTree), "tree is not a KDTree."
    balls = tree.query_ball_tree(tree, radius_px)
    count_in_cells = np.array([np.mean(box.flatten()[ball], axis=0)
                               for ball in balls])

    return count_in_cells


def get_var_in_cells(box, radius_px, L, tree=None):

    if tree is None:
        tree = get_tree(box)
    else:
        assert isinstance(tree, KDTree), "tree is not a KDTree."
    balls = tree.query_ball_tree(tree, radius_px)
    var_in_cells = np.array([np.var(box.flatten()[ball], axis=0)
                             for ball in balls])

    return var_in_cells


def plot_count_in_cells(
    box, radii_px, L, tree=None, counts=None, label="ionisation", cmap="Greys"
):

    ndim = box.ndim
    N = box.shape[0]
    radii_Mpc = np.array(radii_px) * L / N

    if ndim == 2:
        slice_im = box
    elif ndim == 3:
        slice_im = box[..., N // 2]

    if tree is None:
        tree = get_tree(box)
    else:
        assert isinstance(tree, KDTree), "tree is not a KDTree."
    if counts is not None:
        assert len(counts) == len(
            radii_px
        ), "Number of radii inconsistent with number of counts."

    rcmap = truncate_colormap(cm.get_cmap("PuRd"), 0.2, 1.0)
    norm = colors.Normalize(vmin=np.min(radii_Mpc), vmax=np.max(radii_Mpc))

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": (1.2, 1.0)}
    )

    im = axes[1].imshow(slice_im, cmap=cmap, origin="lower",
                        extent=(0, L, 0, L))
    plt.colorbar(im, label=label, fraction=0.045)
    x0 = np.max(radii_Mpc) * 1.5
    d = 5.0
    axes[0].hist(
        box.flatten(),
        histtype="step",
        bins=60,
        lw=1.5,
        ls="--",
        color="k",
        label="Pixel distribution",
        density=True,
        alpha=0.8,
    )
    for ir in range(len(radii_px)):
        # image
        if len(radii_px) < 4:
            coord = x0 + np.sum(2 * radii_Mpc[: ir + 1]) + ir * d
        else:
            coord = x0
        circle = plt.Circle(
            (coord, coord),
            radius=radii_Mpc[ir],
            color=rcmap(norm(radii_Mpc[ir])),
            fill=False,
            lw=3.0,
        )
        axes[1].add_patch(circle)
        # PDF
        if counts is None:
            count_in_cells = get_count_in_cells(box, radii_px[ir], tree=tree)
        else:
            count_in_cells = counts[ir]
        axes[0].hist(
            count_in_cells,
            histtype="step",
            bins=75,
            density=True,
            color=rcmap(norm(radii_Mpc[ir])),
            lw=2.0,
            label=rf"$R={radii_Mpc[ir]:.1f}~\mathrm{{Mpc}}$",
        )
    axes[0].set_xlabel(r"Mean in ball of radius $R$")
    axes[0].set_ylabel("Probability")
    if len(radii_px) < 4:
        axes[0].legend(frameon=False)
    else:
        sm = cm.ScalarMappable(cmap=rcmap, norm=norm)
        plt.colorbar(sm, fraction=0.05, ax=axes[0], label=r"Radius $R$ [Mpc]")

    axes[1].set_xlabel(r"$x$ [Mpc]")
    axes[1].set_ylabel(r"$y$ [Mpc]")
    fig.tight_layout()

    return fig, axes


def plot_mean_counts(
    box,
    L,
    counts,
    radius_px=None,
    label="ionisation",
    cmap="Greys",
    newfig=True,
    fig=None,
    color="k",
):

    ndim = box.ndim
    N = box.shape[0]
    if ndim == 2:
        slice_im = box
    elif ndim == 3:
        slice_im = box[..., N // 2]

    if fig is None:
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": (1.2, 1.0)}
        )
    else:
        fig = plt.gcf()
        axes = fig.axes

    if newfig:
        im = axes[1].imshow(slice_im, cmap=cmap, origin="lower",
                            extent=(0, L, 0, L))
        plt.colorbar(im, label=label, fraction=0.045)
        axes[0].hist(
            box.flatten(),
            histtype="step",
            bins=60,
            lw=1.5,
            ls="--",
            color="k",
            density=True,
            alpha=0.8,
        )

    if radius_px is not None:
        coord = radius_px * L / N * 4.0
        circle = plt.Circle(
            (coord, coord), radius=radius_px * L / N, color=color,
            fill=False, lw=3.0
        )
        axes[1].add_patch(circle)

    pdfs = []
    for count in counts:
        pdf, bin_edges = np.histogram(count, bins=60,
                                      normed=True, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2
        axes[0].plot(bins, pdf,
                     drawstyle="steps-mid", lw=0.8,
                     color=color, alpha=0.2)
        pdfs.append(pdf)
    axes[0].plot(bins, np.mean(pdfs, axis=0), color=color, lw=2.0)
    axes[0].set_xlabel(rf"Mean {label} in ball of radius $R$")
    axes[0].set_ylabel("Probability")
    # axes[0].legend(frameon=False)
    axes[1].set_xlabel(r"$x$ [Mpc]")
    axes[1].set_ylabel(r"$y$ [Mpc]")
    fig.tight_layout()

    return fig, axes


def plot_evolution_count_in_cells(
    radius_px, L, N, counts, redshifts, label="ionisation"
):

    radius_Mpc = np.array(radius_px) * L / N

    assert len(counts) == len(
        redshifts
    ), "Number of radii inconsistent with number of counts."

    zcmap = truncate_colormap(cm.get_cmap("PuRd"), 0.2, 1.0)
    norm = colors.Normalize(vmin=np.min(redshifts), vmax=np.max(redshifts))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    for iz, z in enumerate(redshifts):
        ax.hist(
            counts[iz],
            histtype="step",
            bins=75,
            density=True,
            color=zcmap(norm(z)),
            lw=2.0,
            label=rf"$z={z:.1f}$",
        )
    ax.set_xlabel(rf"Mean {label} in ball of radius $R={radius_Mpc:.1f}$Mpc")
    ax.set_ylabel("Probability")
    if len(redshifts) < 4:
        ax.legend(frameon=False)
    else:
        sm = cm.ScalarMappable(cmap=zcmap, norm=norm)
        plt.colorbar(sm, fraction=0.05, ax=ax, label=r"Resdhift $z$")
    fig.tight_layout()

    return fig, ax


def bubble_mask(x, R, NDIM, DIM):
    # wrapper to handle different dimensionality
    if NDIM == 2:
        return disk_mask(x, R, DIM)
    elif NDIM == 3:
        return sphere_mask(x, R, DIM)


def disk_mask(pos, R, DIM, periodic=False):
    # generates mask corresponding to a 2D ionised disk
    # pos is coordinates of the centre of the bubble
    # R is its radius

    full_struct = np.zeros([DIM, DIM])

    # Creates a disk at centre of smaller structure to avoid generating
    # another whole box: just enough to contain the disk
    structsize = int(2 * R + 6)
    x0 = y0 = int(structsize / 2)
    struct = np.zeros((structsize, structsize))
    x, y = np.indices((structsize, structsize))
    mask = (x - structsize / 2) ** 2 + (
        y - structsize / 2
    ) ** 2 <= R**2  # puts the disk in the middle of new box
    struct[mask] = 1

    # Now work out coordinate shift to move centre to pos
    xmov = [pos[0] - x0, pos[0] + x0]
    ymov = [pos[1] - y0, pos[1] + y0]

    # if struct goes out of the box
    xmin = max(xmov[0], 0)
    xmax = min(xmov[1], DIM)
    ymin = max(ymov[0], 0)
    ymax = min(ymov[1], DIM)

    # periodic boundary conditions
    if periodic:
        if xmov[0] < 0:
            extra_struct = struct[
                0: abs(xmov[0]), abs(min(0, ymov[0])): min(structsize, DIM - ymov[0])
            ]
            full_struct[DIM - abs(xmov[0]): DIM, ymin:ymax] = np.add(
                full_struct[DIM - abs(xmov[0]): DIM, ymin:ymax], extra_struct
            )
        if xmov[1] > DIM:
            extra_struct = struct[
                structsize - (xmov[1] - DIM): structsize,
                abs(min(0, ymov[0])): min(structsize, structsize + DIM - ymov[1]),
            ]
            full_struct[0: xmov[1] - DIM, ymin:ymax] = np.add(
                full_struct[0: xmov[1] - DIM, ymin:ymax], extra_struct
            )
        if ymov[0] < 0:
            extra_struct = struct[
                abs(min(0, xmov[0])): min(structsize, DIM - xmov[0]), 0: abs(ymov[0])
            ]
            full_struct[xmin:xmax, DIM - abs(ymov[0]): DIM] = np.add(
                full_struct[xmin:xmax, DIM - abs(ymov[0]): DIM], extra_struct
            )
        if ymov[1] > DIM:
            extra_struct = struct[
                abs(min(0, xmov[0])): min(structsize, structsize + DIM - xmov[1]),
                structsize - (ymov[1] - DIM): structsize,
            ]
            full_struct[xmin:xmax, 0: ymov[1] - DIM] = np.add(
                full_struct[xmin:xmax, 0: ymov[1] - DIM], extra_struct
            )

    # truncated struct if some part is outside the full struct
    small_struct = struct[
        abs(xmov[0] - xmin): structsize - abs(xmov[1] - xmax),
        abs(ymov[0] - ymin): structsize - abs(ymov[1] - ymax),
    ]
    # add to previous box
    full_struct[xmin:xmax, ymin:ymax] = np.add(
        full_struct[xmin:xmax, ymin:ymax], small_struct
    )

    return full_struct


def sphere_mask(pos, R, DIM, periodic=False):
    # generates mask corresponding to a 3D ionised sphere
    # pos is coordinates of the centre of the bubble
    # R is its radius

    full_struct = np.zeros([DIM, DIM, DIM])

    # Creates a disk at centre of smaller structure to avoid generating another whole box: just enough to contain the disk
    structsize = int(2 * R + 6)
    x0 = y0 = z0 = int(structsize / 2)
    struct = np.zeros((structsize, structsize, structsize))
    x, y, z = np.indices((structsize, structsize, structsize))
    mask = (x - structsize / 2) ** 2 + (y - structsize / 2) ** 2 + (
        z - structsize / 2
    ) ** 2 <= R**2
    struct[mask] = 1

    # Now work out coordinate shift to move centre to pos
    xmov = [pos[0] - x0, pos[0] + x0]
    ymov = [pos[1] - y0, pos[1] + y0]
    zmov = [pos[2] - z0, pos[2] + z0]

    # if struct goes out of the box
    xmin = max(xmov[0], 0)
    xmax = min(xmov[1], DIM)
    ymin = max(ymov[0], 0)
    ymax = min(ymov[1], DIM)
    zmin = max(zmov[0], 0)
    zmax = min(zmov[1], DIM)

    # periodic boundary conditions
    if periodic:
        if xmov[0] < 0:
            extra_struct = struct[
                0: abs(xmov[0]),
                abs(min(0, ymov[0])): min(structsize, DIM - ymov[0]),
                abs(min(0, zmov[0])): min(structsize, DIM - zmov[0]),
            ]
            full_struct[DIM - abs(xmov[0]): DIM, ymin:ymax, zmin:zmax] = np.add(
                full_struct[DIM - abs(xmov[0]): DIM, ymin:ymax, zmin:zmax],
                extra_struct,
            )
        if xmov[1] > DIM:
            extra_struct = struct[
                structsize - (xmov[1] - DIM): structsize,
                abs(min(0, ymov[0])): min(structsize, structsize + DIM - ymov[1]),
                abs(min(0, zmov[0])): min(structsize, structsize + DIM - zmov[1]),
            ]
            full_struct[0: xmov[1] - DIM, ymin:ymax, zmin:zmax] = np.add(
                full_struct[0: xmov[1] - DIM, ymin:ymax, zmin:zmax], extra_struct
            )
        if ymov[0] < 0:
            extra_struct = struct[
                abs(min(0, xmov[0])): min(structsize, DIM - xmov[0]),
                0: abs(ymov[0]),
                abs(min(0, zmov[0])): min(structsize, DIM - zmov[0]),
            ]
            full_struct[xmin:xmax, DIM - abs(ymov[0]): DIM, zmin:zmax] = np.add(
                full_struct[xmin:xmax, DIM - abs(ymov[0]): DIM, zmin:zmax],
                extra_struct,
            )
        if ymov[1] > DIM:
            extra_struct = struct[
                abs(min(0, xmov[0])): min(structsize, structsize + DIM - xmov[1]),
                structsize - (ymov[1] - DIM): structsize,
                abs(min(0, zmov[0])): min(structsize, structsize + DIM - zmov[1]),
            ]
            full_struct[xmin:xmax, 0: ymov[1] - DIM, zmin:zmax] = np.add(
                full_struct[xmin:xmax, 0: ymov[1] - DIM, zmin:zmax], extra_struct
            )
        if zmov[0] < 0:
            extra_struct = struct[
                abs(min(0, xmov[0])): min(structsize, DIM - xmov[0]),
                abs(min(0, ymov[0])): min(structsize, DIM - ymov[0]),
                0: abs(zmov[0]),
            ]
            full_struct[xmin:xmax, ymin:ymax, DIM - abs(zmov[0]): DIM] = np.add(
                full_struct[xmin:xmax, ymin:ymax, DIM - abs(zmov[0]): DIM],
                extra_struct,
            )
        if zmov[1] > DIM:
            extra_struct = struct[
                abs(min(0, xmov[0])): min(structsize, structsize + DIM - xmov[1]),
                abs(min(0, ymov[0])): min(structsize, structsize + DIM - ymov[1]),
                structsize - (zmov[1] - DIM): structsize,
            ]
            full_struct[xmin:xmax, ymin:ymax, 0: zmov[1] - DIM] = np.add(
                full_struct[xmin:xmax, ymin:ymax, 0: zmov[1] - DIM], extra_struct
            )

    # truncated struct if some part is outside the full struct
    small_struct = struct[
        abs(xmov[0] - xmin): structsize - abs(xmov[1] - xmax),
        abs(ymov[0] - ymin): structsize - abs(ymov[1] - ymax),
        abs(zmov[0] - zmin): structsize - abs(zmov[1] - zmax),
    ]  # truncated struct if some part is outside the full struct
    # add to full box
    full_struct[xmin:xmax, ymin:ymax, zmin:zmax] = np.add(
        full_struct[xmin:xmax, ymin:ymax, zmin:zmax], small_struct
    )  # add to previous box in case some intermediate structures overlap

    return full_struct


def get_radii(radius_Mpc, sigR, num_sources, distribution=0):

    """
    Get radius distribution for a given mean (radius_Mpc), standard deviation (sigR).
    Output has length num_sources.
    Parameters:
        radius_Mpc (float): mean radius of bubbles.
        sigR (float): std of radius distribution
        num_sources (int): Length of the distribution
        distribution (int): Gaussian (1), Lognormal (2) or flat (0)
        distribution of bubble radii
    """

    if distribution == 1:
        radius_distri = np.random.normal(loc=radius_Mpc, scale=sigR,
                                         size=num_sources)
    # Lognormal Instance #####################################################
    elif distribution == 2:
        # mean radius
        mean = np.log(radius_Mpc / np.sqrt(1 + sigR / radius_Mpc**2))
        # deviation from mean
        sigma = np.sqrt(np.log(1 + sigR / radius_Mpc**2))
        radius_distri = np.ranodm.lognormal(mean, sigma, size=num_sources)
    # Flat Distribution #######################################################
    elif distribution == 0:
        radius_distri = np.ones(num_sources) * radius_Mpc
    radius_distri_Mpc = np.sort(radius_distri)[::-1]

    return radius_distri_Mpc


def toy_model_distr1(
    coeval,
    num_sources,
    radius_Mpc,
    sigR,
    distribution=0,
    halo_list=None,
    R_eff=None,
    verbose=False,
    use_KDTree=False,
):
    """
    IG: Toy model which assigns a ionzied bubble to a dense region.

    ###################################################################################################
    #For this version we try and assign the bubble sisez based on a distribution.                     #
    #Currently I've included a Gaussian and Lognormal distribution, which have been randomly weighted.#
    ###################################################################################################

    Parameters:
        coeval: coeval output of 21cmfast at redshift z
        num_sources (array): Number of sources.
        radius_Mpc (float): mean radius of bubbles.
        sigR (float): std of radius distribution
        halo_list: (N,4) array made of the positions (3 first columns) and masses (fourth column) of the
            haloes in the field
        R_eff (float): minimum distance between two haloes forming an ionised bubble
    Returns:
        Ionization cube.
    """

    N = coeval.user_params.HII_DIM  # pixel number of HII field
    L = coeval.user_params.BOX_LEN  # box length in Mpc
    z = coeval.redshift  # redshift considered
    # if R_eff is None:
    #     R_eff = radius_Mpc/2.

    # length of radius distribution
    # larger than num_sources to be able to remove negative values and still have enough
    if halo_list is not None and num_sources is None:
        num_sources = halo_list.shape[0]
    ntemp = int(3.0 * num_sources)

    #####################
    # Get halo positions
    #####################

    if halo_list is None:
        # Assign haloes to the num_sources densest points in the field
        dens = (1.0 + z) ** 3 * (coeval.density + 1.0)  # density field
        dens_flatn = dens.flatten()
        arg_ascend = np.argsort(dens_flatn)[::-1]
        # extract positions of highest density points
        out = np.zeros(dens_flatn.shape)
        out[arg_ascend[:ntemp]] = 1
        xf_out = out.reshape(dens.shape)
        pos = np.argwhere(xf_out == 1.0)
        # sort position with decreasing density
        dens_list = np.array([dens[tuple(p)] for p in pos])
        pos = pos[np.argsort(dens_list)[::-1], :]
        dens_list = np.sort(dens_list)[::-1]
    else:
        # Assign bubble to haloes from halo list
        # Largest radii assigned to most massive haloes
        dens_list = halo_list[:, -1]
        pos = np.array(halo_list[:, :-1], dtype=int)

        # # sort haloes and radii by decreasing order (mass, size respectively)
        # arg_ascend = np.flip(np.argsort(hmasses))
        # dens_list = hmasses[arg_ascend][:ntemp]
        # pos = pos[arg_ascend, :][:ntemp, :]

    # remove density peaks that are closer than Reff
    # equivalent to FoF algorithm
    pos_Mpc = pos * L / N
    if R_eff is not None:
        if verbose:
            print("Removing density peaks closer than Reff...")
        # i = 0
        # while i < pos_Mpc.shape[0]:
        #     p = pos_Mpc[i]
        #     mask = np.ones(pos_Mpc.shape[0], dtype=bool)
        #     mask[i+1:][np.where(np.linalg.norm(p-pos_Mpc[i+1:], axis=1) < R_eff)] = False
        #     pos_Mpc = pos_Mpc[mask]
        #     pos = pos[mask]
        #     dens_list = dens_list[mask]
        #     i += 1
        tree = KDTree(pos_Mpc)
        # find halos separated by less than R_eff Mpc
        balls = tree.query_ball_tree(tree, R_eff)
        # average positions to find centre of halo group
        pos = np.array([np.mean(pos[ball], axis=0) for ball in balls],
                       dtype=int)
        # average masses around designated centre
        dens_peaks = np.array([np.mean(dens_list[ball], axis=0)
                               for ball in balls])
        # identify overlapping centres and remove mutiplets
        temp, ind_unique = np.unique(
            np.ravel_multi_index(pos.T, (N, N, N)), return_index=True
        )
        pos = pos[ind_unique]
        dens_list = dens_peaks[ind_unique]
        # sort by decreasing halo mass
        arg_ascend = np.flip(np.argsort(dens_list))
        dens_list = dens_list[arg_ascend]
        pos = pos[arg_ascend, :]

    if pos_Mpc.shape[0] < num_sources and halo_list is None:
        warnings.warn(
            "Low number of bubbles, lowering the Reff parameter is recommended."
        )
    else:
        pos = pos[:num_sources, :]
        pos_Mpc = pos_Mpc[:num_sources, :]
        dens_list = dens_list[:num_sources]
    num_sources = min(pos_Mpc.shape[0], num_sources)

    ##########################
    # Get radius distribution
    ##########################

    radius_distri_Mpc = get_radii(radius_Mpc, sigR, ntemp, distribution)
    radius_distri = np.array(radius_distri_Mpc * N / L, dtype=int)
    # remove negative radii which can appear if the mean of the distribution is close to zero
    if np.any(radius_distri < 0):
        radius_distri_Mpc = radius_distri_Mpc[radius_distri >= 0]
        radius_distri = radius_distri[radius_distri >= 0]
        warnings.warn(
            "Negative values were removed. "
            "This can cause issues if there were too many."
        )
    assert (
        len(radius_distri) >= num_sources
    ), "Error: too many negative radii, need to increase num_sources"

    # stochastically select num_sources elements in the remaining distribution
    # and sort them by decreasing order
    inds = np.random.choice(
        np.arange(0, radius_distri.size), size=num_sources, replace=False
    )  # random indices with no repeat
    radius_distri = np.sort(radius_distri[inds])[::-1]
    radius_distri_Mpc = np.sort(radius_distri_Mpc[inds])[::-1]

    # Position sources in densest locations or on located haloes.
    # Loop will stop either when all dense points have been populated
    # or when the ionised fraction for the toy model has reached the one
    # of the coeval cube

    if not use_KDTree:
        true_pos, true_radii = [], []
        box = np.zeros((N, N, N))
        i = 0
        while (len(true_radii) < num_sources) and (
            box.mean() < (1.0 - coeval.xH_box.mean())
        ):

            R = radius_distri[i]  # nd]#//(nrad//num_sources)]
            if R < 1:
                box[tuple(pos[i])] = 1
            else:
                bmask = bubble_mask(pos[i], R, NDIM=3, DIM=N)
                box = np.add(box, bmask).astype(bool)
                box = box.astype(int)

            true_pos.append(pos[i])
            true_radii.append(R)
            i += 1

            if verbose:
                sys.stdout.write(
                    f"\r{int(len(true_radii) / num_sources * 100)}% done"
                    )
                sys.stdout.flush()

        nbubbles = len(true_radii)
        if verbose:
            sys.stdout.write("\r%i bubbles, %i%% done\n" % (nbubbles, 100))
            sys.stdout.flush()
    else:
        x, y, z = np.mgrid[0:N, 0:N, 0:N]
        points = np.c_[x.ravel(), y.ravel(), z.ravel()]
        tree = KDTree(points)

        box = np.zeros((N, N, N))
        for R in np.unique(radius_distri):
            if box.mean() < (1.0 - coeval.xH_box.mean()):
                temp_tree = KDTree(pos[radius_distri == R, :])
                balls = tree.query_ball_tree(temp_tree, R)
                box[np.unravel_index(sum(balls, []), (N, N, N))] = 1.0
            else:
                break

    return box, nbubbles, np.array(true_pos), np.array(true_radii), np.array(dens_list)


def distri2box(distri, N, NDIM=3, force_mean=False):

    from itertools import permutations

    poss_pos = np.array(
        [i for i in permutations(np.arange(N), 3)]
    )  # all possible positions
    box = np.zeros((N, N, N))
    distri = np.array(distri, dtype=int)
    nbubbles = len(distri)
    pos = poss_pos[
        np.random.choice(np.arange(poss_pos.shape[0]), size=nbubbles,
                         replace=False), :
    ]

    box = np.zeros((N, N, N))
    for i in range(nbubbles):

        R = distri[i]
        #         print(R,pos[i])
        if R < 1:
            box[tuple(pos[i])] = 1
        else:
            bmask = bubble_mask(pos[i], R, NDIM=3, DIM=N)
            box = np.add(box, bmask).astype(bool)
            box = box.astype(int)

        if force_mean is not None and box.mean() >= float(force_mean):
            break

        sys.stdout.write("\r%i%% done" % ((i + 1) / nbubbles * 100))
        sys.stdout.flush()

    sys.stdout.write("\r%i bubbles, %i%% done\n" % (i, 100))
    sys.stdout.flush()

    return box, pos


def lin_dens_contrast(rho, nu=21.0 / 13.0):
    return nu * (1.0 - rho ** (-1.0 / nu))


def init_power(k, As=10, ns=-1.5):
    return As * k ** (ns)


def sigma_L(R, z, k=np.logspace(-3, 3, 500),
            cos=cosmology.Planck15, **kwargs):
    
    pars = camb.CAMBparams(WantTransfer=True, NonLinear="NonLinear_both")
    pars.set_cosmology(
        H0=cos.H0.value, ombh2=cos.Ob0 * cos.h**2,
        omch2=cos.Odm0 * cos.h**2
    )
    pars.InitPower.set_params(ns=0.9667, As=2.142e-9)
    pars.set_matter_power(redshifts=[z], kmax=1000.)
    results = camb.get_results(pars)
    sigmaL = results.get_sigmaR(
        R, z_indices=[0], hubble_units=True,
        var1="delta_tot", var2="delta_tot"
    )[0]

    return sigmaL


def psi(rho, R, z, cos=cosmology.Planck15, **kwargs):

    sigma_LR = sigma_L(R, z, cos=cos, **kwargs)
    sigma_Lrho = np.array(
        [sigma_L(R * r ** (1 / 3), z, cos=cos, **kwargs)
         for r in np.atleast_1d(rho)]
    )
    return lin_dens_contrast(rho) ** 2 * sigma_LR**2 / 2.0 / sigma_Lrho**2


def get_theory_cic(rhom_range, sigma_mu, R, z, use_camb=False, **kwargs):

    assert np.size(rhom_range) > 2
    rhom_range = np.atleast_1d(rhom_range)

    # define differential ranges
    drho = np.diff(rhom_range).mean()
    rhomin = rhom_range.min()
    rhomax = rhom_range.max()
    rhom_range_diff = np.arange(rhomin-drho/2, rhomax+drho, step=drho)
    rhom_range_diff2 = np.arange(rhomin-drho, rhomax+drho*1.5, step=drho)

    assert (rhom_range.min() > 0.) and (rhom_range_diff.min() > 0.) \
        and (rhom_range_diff2.min() > 0.), \
        "Psi can only be computed on non-zero values of rhom."

    # compute exponential decay of the pdf
    # and its derivatives
    psi_rhom = psi(rhom_range, R, z, use_camb=use_camb, **kwargs)
    diff1 = np.diff(psi(rhom_range_diff, R, z, use_camb=use_camb, **kwargs))/drho
    diff2 = np.diff(psi(rhom_range_diff2, R, z, use_camb=use_camb, **kwargs), n=2)/drho**2

    # mask nan and inf values if any
    mask = (np.isnan(diff1) + np.isnan(diff2) + np.isnan(psi_rhom))
    mask += np.isinf(diff1) + np.isinf(diff2) + np.isinf(psi_rhom)
    if mask.any():
        warnings.warn('NaNs in your derivatives...')

    # probability distribution
    PR = np.sqrt((diff2+diff1/rhom_range)/(2.*np.pi*sigma_mu**2)) \
        * np.exp(-psi_rhom/sigma_mu**2)

    return PR


def get_normalised_cic(rhom_range, sigma_mu, R, z, use_camb=False, **kwargs):

    PR = get_theory_cic(rhom_range, sigma_mu, R, z, use_camb=use_camb, **kwargs)
    mean_PR = np.trapz(PR, rhom_range)
    mean_rho = np.trapz(rhom_range*PR, rhom_range)
    est_PR = (mean_rho / mean_PR**2) \
             * get_theory_cic(rhom_range * mean_rho / mean_PR, sigma_mu, R, z, use_camb=use_camb, **kwargs)
    norm_PR = est_PR / np.trapz(est_PR, rhom_range)

    return norm_PR
