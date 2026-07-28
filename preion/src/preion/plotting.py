#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["corner", "hist2d", "error_ellipse"]
__version__ = "0.0.6"
__author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
__copyright__ = "Copyright 2013 Daniel Foreman-Mackey"
__contributors__ = [
    # Alphabetical by first name.
    "Adrian Price-Whelan @adrn",
    "Brendon Brewer @eggplantbren",
    "Ekta Patel @ekta1224",
    "Emily Rice @emilurice",
    "Geoff Ryan @geoffryan",
    "Kyle Barbary @kbarbary",
    "Phil Marshall @drphilmarshall",
    "Pierre Gratier @pirg", " MD " 
]

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import scipy.stats
import scipy.ndimage as nd
import matplotlib 


def corner(xs, weights=None, labels=None, extents=None, truths=None,
           truth_color="#4682b4", scale_hist=False, quantiles=[], plot_hist=True,
           verbose=False, plot_contours=True, plot_datapoints=True, corrcoef=False,
           fill_contours=True,
           fig=None,sigmas=[1,2,3], smooth=0,alpha=.5, zorder=0, axes=None, **kwargs):
    """
    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    `matplotlib` styling.

    Parameters
    ----------
    xs : array_like (nsamples, ndim)
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    weights : array_like (nsamples,)
        The weight of each sample. If `None` (default), samples are given
        equal weight.

    labels : iterable (ndim,) (optional)
        A list of names for the dimensions.

    extents : iterable (ndim,) (optional)
        A list where each element is either a length 2 tuple containing
        lower and upper bounds (extents) or a float in range (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

    truths : iterable (ndim,) (optional)
        A list of reference values to indicate on the plots.

    truth_color : str (optional)
        A ``matplotlib`` style color for the ``truths`` makers.

    scale_hist : bool (optional)
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    quantiles : iterable (optional)
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    verbose : bool (optional)
        If true, print the values of the computed quantiles.

    plot_contours : bool (optional)
        Draw contours for dense regions of the plot.

    plot_datapoints : bool (optional)
        Draw the individual data points.

    fig : matplotlib.Figure (optional)
        Overplot onto the provided figure object.

    """

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError('weights must be 1-D')
        if xs.shape[1] != weights.shape[0]:
            raise ValueError('lengths of weights must match number of samples')

    # backwards-compatibility
    plot_contours = kwargs.get("smooth", plot_contours)

    K = len(xs)
    factor = 3.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin
    trdim = 0.05 * factor  # size of top/right margin
    whspace = 0.2        # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        if axes is not None:
            assert axes.shape == (K, K)
        else:
            try:
                axes = np.array(fig.axes).reshape((K, K))
            except:
                raise ValueError("Provided figure has {0} axes, but data has "
                                 "dimensions K={1}".format(len(fig.axes), K))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    if axes is None:
        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)

    if extents is None:
        extents = [[x.min(), x.max()] for x in xs]

        # Check for parameters that never change.
        m = np.array([e[0] == e[1] for e in extents], dtype=bool)
        if np.any(m):
            raise ValueError(("It looks like the parameter(s) in column(s) "
                              "{0} have no dynamic range. Please provide an "
                              "`extent` argument.")
                             .format(", ".join(map("{0}".format,
                                                   np.arange(len(m))[m]))))
    else:
        # If any of the extents are percentiles, convert them to ranges.
        for i in range(len(extents)):
            try:
                emin, emax = extents[i]
            except TypeError:
                q = [0.5 - 0.5*extents[i], 0.5 + 0.5*extents[i]]
                extents[i] = quantile(xs[i], q, weights=weights)

    # max_prob = np.zeros(len(xs))
    for i, x in enumerate(xs):
        if not np.any(x):
            continue

        ax = axes[i, i]
        # Plot the histograms.
        if plot_hist:
            n,b = hist1d( x, weights, smooth=smooth, ax=ax, extents=extents[i], zorder=zorder, **kwargs)
        # max_prob[i] = np.max(n)
        # print(max_prob)
        if truths is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        ls_quantiles = ['--' for q in quantiles]
        if (0.5 in quantiles):
            ls_quantiles[quantiles.index(0.5)]= '-'
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for u,q in enumerate(qvalues):
                ax.axvline(q, ls=ls_quantiles[u], color=kwargs.get("color", "k"),lw=1.)
            if verbose:
                print("Quantiles:")
                print(['%s: %.4f' %(quantiles[u], qvalues[u]) for u in range(len(quantiles))])

        # Set up the axes.
        ax.set_xlim(extents[i])
        if scale_hist:
            ax.set_ylim(-0.1 * np.max(n), 1.1 * np.max(n))
        else:
            ax.relim()
            ax.autoscale(axis='y')
            ax.set_ylim(bottom=0)
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))

        # Not so DRY.
        if i < K - 1:
            ax.set_xticklabels([])
        else:
            [l.set_rotation(90) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i])
#                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs):
            if not np.any(y):
                continue
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue

            hist2d(y, x, ax=ax, extent=[extents[j], extents[i]],
                   plot_contours=plot_contours,
                   plot_datapoints=plot_datapoints,
                   sigmas = sigmas, corrcoef=corrcoef,
                   weights=weights, smooth=smooth, fill_contours=fill_contours,
                   alpha=alpha, zorder=zorder, **kwargs)

            if truths is not None:
                # ax.plot(truths[j], truths[i], "s", color=truth_color)
                ax.axvline(truths[j], color=truth_color)
                ax.axhline(truths[i], color=truth_color)

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(90) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j])
#                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                #[l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i])
#                    ax.yaxis.set_label_coords(-0.3, 0.5)

    return fig


def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
                          width=2 * np.sqrt(S[0]) * factor,
                          height=2 * np.sqrt(S[1]) * factor,
                          angle=theta,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = pl.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot


def hist2d(x, y, smooth=0., sigmas=[1,2,3], corrcoef=False, color='k', fill_contours=True, *args, **kwargs):
    """
    Plot a 2-D histogram of samples.

    """
    ax = kwargs.pop("ax", pl.gca())

    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 50)
    # color = kwargs.pop("color", "k")
    alpha = kwargs.pop("alpha", 0.8)
    linewidths = kwargs.pop("linewidths", None)
    plot_datapoints = kwargs.get("plot_datapoints", True)
    plot_contours = kwargs.get("plot_contours", True)
    weights = kwargs.get('weights', None)
    zorder = kwargs.get('zorder',0)
#MD
    cmap  = matplotlib.colormaps[kwargs.get("cmap", "gray")]
    # labels=[r'$%i-\sigma$' %(cr) for cr in sigmas]
    # levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

#    cmap = cm.get_cmap("gray")
#    cmap._init()
#    cmap._lut[:-3, :-1] = 0.
#    cmap._lut[:-3, -1] = np.linspace(1, 0, cmap.N)
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 weights=weights,range=list(extent))
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "`extent` argument.")

    # print(H)
    H = nd.gaussian_filter(H, smooth)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    levels = 1.-(np.exp(-0.5 * np.array(sigmas) ** 2))
    # levels = levels[::-1]
    # print(levels) 
    V = np.empty(len(levels))

    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        raise RuntimeWarning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # compute the bin centres
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    # X, Y = X[:-1], Y[:-1]

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )
#MD 
    # gkde = scipy.stats.gaussian_kde([x, y])
    # dx   = (extent[0][1]-extent[0][0])/(bins+1.)
    # dy   = (extent[1][1]-extent[1][0])/(bins+1.)
    
    
    # x2,y2 = np.mgrid[extent[0][0]:extent[0][1]:dx, extent[1][0]:extent[1][1]:dy]
    # z   = np.array(gkde.evaluate([x2.flatten(),y2.flatten()])).reshape(x2.shape)


    # W=(np.exp(-0.5 * np.array(sigmas) ** 2))*z.max()
    # W=np.append(W[::-1],z.max())
    
    # WW = (ctr_level(z, contours, infinite=False))[:3]
    # WW=WW[::-1]
      
    if plot_datapoints:
        ax.plot(x, y, "o", color=color, ms=1.5, zorder=-1, alpha=0.1,rasterized=True)
    
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(levels))
    if fill_contours:
        colors = list([cmap(level) for level in levels])
    else:
        colors = [color for level in levels]
    # print(V)
    if plot_contours:
#        ax.pcolor(X, Y, H.max() - H.T, cmap=cmap)
#        ax.contour(X1, Y1, H.T, V, colors=color, linewidths=linewidths)
        # ax.contourf(x2, y2, z, W, alpha=alpha, cmap=cmap, antialiased=True)
        # CS = ax.contour(x2, y2, z, W[:np.size(sigmas)-1],linewidths=linewidths, colors=color, antialiased=False)
        if fill_contours:
            ax.contourf(X2, Y2, H2.T, np.append(V,H.max()), alpha=alpha,colors=colors, antialiased=False,zorder=zorder)
        ax.contour(X2, Y2, H2.T, V,linewidths=linewidths, colors=colors, antialiased=False,alpha=alpha,zorder=zorder)
 
        # fmt={}
        # for l,s in zip(contours, labels):
        #     fmt[l]=s
        # ax.clabel(CS,CS.levels,fmt=fmt,inline=True)

    data = np.vstack([x, y])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    if kwargs.pop("plot_ellipse", False):
        error_ellipse(mu, cov, ax=ax, edgecolor="r", ls="dashed")

    if corrcoef:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        C = np.corrcoef(x.flatten(), y.flatten())[0, 1]
        ax.text(0.95, 0.95, fr'$r={C:.3f}$', color=color, #'Correlation {:.3f}'.format(C),
                ha='right', va='top', 
                # bbox=props, 
                transform=ax.transAxes)

def hist1d(x, weights=None, smooth=0., normmax=False, **kwargs):
    ax = kwargs.get("ax", pl.gca())
    bins = kwargs.get("bins", 50)
    extents = kwargs.get("extents",[x.min(),x.max()])
    zorder=kwargs.get("zorder",1)

#    n, b, p = ax.hist(x, weights=weights, bins=kwargs.get("bins", 50),
#                      range=extents, histtype="step",
#                      color=kwargs.get("color", "k"),lw=kwargs.get("lw",1),density=True)
    n, b = np.histogram(x, weights=weights, bins=bins, density=True, range=extents)
    if smooth>0:
        ns = nd.gaussian_filter1d(n,smooth)
    else:
        ns = n
    if normmax:
        ns /= ns.max()
    ax.plot( (b[:-1]+b[1:])/2., ns,
          color=kwargs.get("color", "k"),lw=kwargs.get("lw",1),zorder=zorder,ls=kwargs.get("ls",'-'),label=kwargs.get("label",''))
    ax.set_xlim(extents)

#    gkde = scipy.stats.gaussian_kde( x)
#    dx = (extents[1]-extents[0])/(bins+1.)
#    b = np.mgrid[extents[0]:extents[1]:dx]
#    n = np.array(gkde.evaluate(b.flatten())).reshape(b.shape)
#    n = n/n.max()
#    ax.plot( b, n, color=kwargs.get("color", "k"),lw=kwargs.get("lw",1))
#    ax.set_xlim(extents)

    return (ns,b)


def ctr_level(histogram2d, lvl, infinite=False):
    """
    Extract the contours for the 2d plots (Karim Benabed)
    """

    hist = histogram2d.flatten()*1.
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist /= cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)[::-1]
    clist = [0]+[hist[-i] for i in alvl]+[hist.max()]
    if not infinite:
        return clist[1:]
    return clist

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

