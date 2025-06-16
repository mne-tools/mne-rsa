"""Functions related to visualization of RDMs.

Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
Takao Shimizu <tshimizu.midd@gmail.com>
"""

import types
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from mne.viz import Brain
from mne.viz.topo import _iter_topography
from scipy.spatial import distance


def plot_rdms(rdms, names=None, items=None, n_rows=1, cmap="viridis", title=None):
    """Plot one or more RDMs.

    Parameters
    ----------
    rdms : ndarray | list of ndarray
        The RDM or list of RDMs to plot. The RDMs can either be two-dimensional (n_items
        x n_items) matrices or be in condensed form.
    names : str | list of str | None
        For each given RDM, a name to show above it. Defaults to no names.
    items : list of str | None
        The each item (row/col) in the RDM, a string description. This will be displayed
        along the axes. Defaults to None which means the items will be numbered.
    n_rows : int
        Number of rows to use when plotting multiple RDMs at once. Defaults to 1.
    cmap : str
        Matplotlib colormap to use. See
        https://matplotlib.org/gallery/color/colormap_reference.html
        for all possibilities. Defaults to 'viridis'.
    title : str | None
        Title for the entire figure. Defaults to no title.

    Returns
    -------
    fig : matplotlib figure
        The figure produced by matplotlib

    """
    if not isinstance(rdms, list):
        rdms = [rdms]

    if isinstance(names, str):
        names = [names]
    if names is not None and len(names) != len(rdms):
        raise ValueError(
            f"Number of given names ({len(names)}) does not "
            f"match the number of RDMs ({len(rdms)})"
        )

    n_cols = int(np.ceil(len(rdms) / n_rows))
    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    ax = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * n_cols + col % n_cols
            if i < len(rdms):
                rdm = rdms[i]
                if rdm.ndim == 1:
                    rdm = distance.squareform(rdm)
                elif rdm.ndim > 2:
                    raise ValueError(f"Invalid shape {rdm.shape} for RDM")
                im = ax[row, col].imshow(rdm, cmap=cmap)

                if names is not None:
                    name = names[i]
                    ax[row, col].set_title(name)
                if items is not None:
                    ax[row, col].set_xticks(np.arange(len(items)))
                    ax[row, col].set_xticklabels(items)
                    ax[row, col].set_yticks(np.arange(len(items)))
                    ax[row, col].set_yticklabels(items)
            else:
                ax[row, col].set_visible(False)

    plt.colorbar(im, ax=ax)
    if title is not None:
        plt.suptitle(title)
    return fig


def _click_func(ax, ch_idx, rdms, cmap):
    """Plot a single RDM interactively.

    Parameters
    ----------
    ax: matplotlib.Axes.axes
        Axes.axes object on which a new single RDM is plotted.
    ch_idx: int
        Index of a channel.
    rdms: ndarray, shape (n_sensors, n_rdm_datapoint)
        RDMs of MEG recordings; there's one RDM for each sensor.
    cmap: str
        Colormap used for plotting RDMs. Check ``matplotlib.pyplot.imshow`` for details.

    """
    rdm = rdms[ch_idx]
    rdm = distance.squareform(rdm)
    ax.imshow(rdm, cmap=cmap)


def plot_rdms_topo(
    rdms,
    info,
    time=None,
    layout=None,
    fig=None,
    axis_facecolor="w",
    axis_spinecolor="w",
    fig_facecolor="w",
    figsize=(6.4, 4.8),
    cmap="viridis",
    show=True,
):
    """Plot RDMs on 2D sensor topography.

    Parameters
    ----------
    rdms: ndarray, shape (n_sensors,[ n_times,] n_rdm_datapts)
        RDMs of MEG/EEG recordings; one RDM for each sensor and time point.
    info: mne.io.meas_info.Info
        Info object that contains meta data of MEG/EEG recordings.
    time: int | [int, int] | None
        A time point (int) or time window ([int, int]) for which RDMs are plotted. When
        a time window is given, averge RDMs for the window are plotted. The default
        (``None``) plots the average RDMs of all the time points. Start of the time
        window is inclusive, while the end is exclusive.
    layout: mne.channels.layout.Layout, optional
        Layout objects containing sensor layout info. The default, ``layout=None``, will
        figure out layout based on info.
    fig: matplotlib.pyplot.Figure | None, optional
        Figure object on which RDMs on 2D sensor topography are plotted. The default
        (``None``) creates a new Figure object with a title based on time parameter.
    axis_facecolor: str, optional
        Face color of the each RDM. Defaults to 'w', white.
    axis_spinecolor: str, optional
        Spine color of each RDM. Defaults to 'w', white.
    fig_facecolor: str, optional
        Face color of the entire topography. Defaults to 'w', white.
    figsize: tuple of float, optional
        Figure size. The first element specify width and the second height.
        Defaults to (6.4, 4.8).
    cmap: str, optional
        Colormap used for plotting RDMs. Defaults to 'viridis'.
        Check :func:`matplotlib.pyplot.imshow` for details.
    show: bool, optional
        Whether to display the generated figure. Defaults to ``True``.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Figure object in which RDMs are plotted on 2D sensor topography.

    """
    if isinstance(rdms, types.GeneratorType):
        rdms = np.array(list(rdms))

    if rdms.ndim != 2 and rdms.ndim != 3:
        raise ValueError(
            "RDMs have to be a 2D or 3D ndarray, "
            "[n_sensors,[ n_times,] n_rdm_datapoints]"
        )
    if len(rdms.shape) == 2:
        rdms = rdms[:, np.newaxis, :]
    if time is None:
        time = [0, rdms.shape[1]]
    if isinstance(time, int):
        time = [time, time + 1]
    if not isinstance(time, list):
        raise TypeError("`time` has to be int, list of [int, int] or None.")
    if (not all(isinstance(i, int) for i in time)) or (len(time) != 2):
        raise TypeError("`time` has to be int, list of [int, int] or None.")
    if time[0] >= time[1]:
        raise ValueError(
            "The start of the time window has to be smaller "
            "than the end of the time window."
        )
    if time[0] < 0 or time[1] > rdms.shape[1]:
        raise ValueError(
            "The time window is out of range. The minimum is 0 "
            f"and the maximum is {rdms.shape[1]}"
        )
    if (fig is not None) and (not isinstance(fig, plt.Figure)):
        raise TypeError("fig has to be matplotlib.pyplot.Figure or None.")

    rdms_cropped = rdms[:, time[0] : time[1], :]
    rdms_avg = rdms_cropped.mean(axis=1)
    # Set title to time window
    if time[0] + 1 != time[1]:
        title = f"From {time[0]} (inclusive) to {time[1]} (exclusive)"
    else:
        title = f"Time point: {time[0]}"

    on_pick = partial(_click_func, rdms=rdms_avg, cmap=cmap)

    if fig is None:
        fig = plt.figure(figsize=figsize)
        if title is not None:
            fig.suptitle(title, x=0.98, horizontalalignment="right")

    my_topo_plot = _iter_topography(
        info=info,
        layout=layout,
        on_pick=on_pick,
        fig=fig,
        axis_facecolor=axis_facecolor,
        axis_spinecolor=axis_spinecolor,
        fig_facecolor=fig_facecolor,
        unified=False,
    )

    for i, (ax, _) in enumerate(my_topo_plot):
        rdms_i = rdms_avg[i]
        rdms_i = distance.squareform(rdms_i)
        ax.imshow(rdms_i, cmap=cmap)

    if show:
        fig.show()

    return fig


def plot_roi_map(values, rois, subject, subjects_dir, cmap="plasma", alpha=1.0):
    """Plot ROI values on a FreeSurfer brain.

    Parameters
    ----------
    values : array-like, shape (n_rois,)
        The values to plot. One value per ROI.
    rois : list of mne.Label
        The labels corresponding to the ROIs.
    subject : str
        The name of the FreeSurfer subject to plot the brain for.
    subjects_dir : str
        The folder in which the FreeSurfer subject data is kept. Inside this folder
        should be a folder with the same name as the `subject` parameter.
    cmap : str
        The name of the matplotlib colormap to use. Defaults to 'plasma'.
    alpha : float
        The alpha (opacity, 1.0 is fully opaque, 0.0 is fully transparent) of the data
        being plotted on top of the brain.

    Returns
    -------
    brain : mne.viz.Brain
        The MNE-Python brain plotting object that was created and currently
        being shown. You can use this to modify the plot.

    """
    cmap = plt.get_cmap(cmap)
    max_val = np.max(values)
    brain = Brain(
        subject=subject, subjects_dir=subjects_dir, surf="inflated", hemi="both"
    )
    labels_lh = np.zeros(len(brain.geo["lh"].coords), dtype=int)
    labels_rh = np.zeros(len(brain.geo["rh"].coords), dtype=int)
    ctab_lh = list()
    ctab_rh = list()
    for i, (roi, value) in enumerate(zip(rois, values), 1):
        if roi.hemi == "lh":
            labels = labels_lh
            ctab = ctab_lh
        else:
            labels = labels_rh
            ctab = ctab_rh
        labels[roi.vertices] = i
        ctab.append([int(x * 255) for x in cmap(value / max_val)[:4]] + [i])
    ctab_lh = np.array(ctab_lh)
    ctab_rh = np.array(ctab_rh)
    brain.add_annotation(
        [(labels_lh, ctab_lh), (labels_rh, ctab_rh)], borders=False, alpha=alpha
    )
    return brain
