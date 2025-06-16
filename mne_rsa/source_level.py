# encoding: utf-8
"""Module implementing representational similarity analysis (RSA) at the source level.

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational
similarity analysis - connecting the branches of systems neuroscience.
Frontiers in Systems Neuroscience, 2(November), 4.
https://doi.org/10.3389/neuro.06.004.2008

Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
Annika Hultén <annika.hulten@aalto.fi>
Ossi Lehtonen <ossi.lehtonen@aalto.fi>
"""

from copy import deepcopy
from warnings import warn

import mne
import nibabel as nib
import numpy as np
from mne.utils import logger, verbose
from scipy.linalg import block_diag

from .rdm import _n_items_from_rdm, rdm_array
from .rsa import rsa_array
from .searchlight import searchlight
from .sensor_level import _construct_tmin, _tmin_tmax_to_indices


@verbose
def rsa_stcs(
    stcs,
    rdm_model,
    src=None,
    spatial_radius=None,
    temporal_radius=None,
    stc_rdm_metric="correlation",
    stc_rdm_params=dict(),
    rsa_metric="spearman",
    ignore_nan=False,
    y=None,
    labels_stcs=None,
    labels_rdm_model=None,
    n_folds=1,
    sel_vertices=None,
    sel_vertices_by_index=None,
    tmin=None,
    tmax=None,
    n_jobs=1,
    verbose=False,
):
    """Perform RSA in a searchlight pattern on MNE-Python source estimates.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computed for a patch surrounding the source point. Source estimate
    objects can be either defined along a cortical surface (``SourceEstimate``
    objects) or volumetric (``VolSourceEstimate`` objects).  For surface source
    estimates, distances between vertices are measured in 2D space, namely as
    the length of the path along the surface from one vertex to another. For
    volume source estimates, distances are measured in 3D space as a straight
    line from one voxel to another.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate | list of mne.VolSourceEstimate
        For each item, a source estimate for the brain activity.
    rdm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model RDM, see :func:`compute_rdm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model RDMs.

        Use :func:`compute_rdm` to compute RDMs.
    src : instance of mne.SourceSpaces | None
        The source space used by the source estimates specified in the ``stcs``
        parameter. Only needed when ``spatial_radius`` is used to create spatial
        searchlight patches. Defaults to None.
    spatial_radius : float | None
        The spatial radius of the searchlight patch in meters. All source points within
        this radius will belong to the searchlight patch. When this is set, ``src`` also
        must be set to the correct source space. Set to None to only perform the
        searchlight over time, flattening across sensors. Defaults to None.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to None.
    stc_rdm_metric : str
        The metric to use to compute the RDM for the source estimates. This can
        be any metric supported by the scipy.distance.pdist function. See also
        the ``stc_rdm_params`` parameter to specify and additional parameter
        for the distance function. Defaults to 'correlation'.
    stc_rdm_params : dict
        Extra arguments for the distance metric used to compute the RDMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    rsa_metric : str
        The RSA metric to use to compare the RDMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    y : ndarray of int, shape (n_items,) | None
        Deprecated, use ``labels_stcs`` and ``labels_rdm_model`` instead.
        For each source estimate, a number indicating the item to which it
        belongs. Defaults to ``None``, in which case ``labels_stcs`` is used.
    labels_stcs : list | None
        For each source estimate, a label that identifies the item to which it
        corresponds. This is used in combination with ``labels_rdm_model`` to align the
        data and model RDMs before comparing them. Multiple source estimates may
        correspond to the same item, in which case they should have the same label and
        will either be averaged when computing the data RDM (``n_folds=1``) or used for
        cross-validation (``n_folds>1``). Labels may be of any python type that can be
        compared with ``==`` (int, float, string, tuple, etc). By default (``None``),
        the integers ``0:len(evokeds)`` are used as labels.

        .. versionadded:: 0.10
    labels_rdm_model: list | None
        For each row in ``rdm_model``, a label that identifies the item to which it
        corresponds. This is used in combination with ``labels_stcs`` to align the
        data and model RDMs before comparing them. Each row should have a unique label.
        Labels may be of any python type that can be compared with ``==`` (int, float,
        string, tuple, etc). By default (``None``), the integers ``0:n_rows`` are used
        as labels.

        .. versionadded:: 0.10
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    sel_vertices :  mne.Label | list of mne.Label | list of ndarray | None
        When set, searchlight patches will only be generated for the subset of
        ROI labels, or vertices/voxels with the given vertex numbers. When
        giving vertex numbers, supply a list of numpy arrays with for each
        hemisphere, the selected vertex numbers. For volume source spaces,
        supple a list with only a single element, with that element being the
        ndarray with vertex numbers.
        See also ``sel_vertices_by_index`` for an alternative manner of
        selecting vertices.
    sel_vertices_by_index : ndarray of int, shape (n_vertices,)
        When set, searchlight patches will only be generated for the subset of
        vertices with the given indices in the ``stc.data`` array.
        See also ``sel_vertices`` for an alternative manner of selecting vertices.
    tmin : float | None
        When set, searchlight patches will only be generated from subsequent
        time points starting from this time point. This value is given in
        seconds. Defaults to ``None``, in which case patches are generated
        starting from the first time point.
    tmax : float | None
        When set, searchlight patches will only be generated up to and
        including this time point. This value is given in seconds. Defaults to
        ``None``, in which case patches are generated up to and including the
        last time point.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate | list of SourceEstimate | list of VolSourceEstimate | float | ndarray
        The correlation values for each searchlight patch. When spatial_radius
        None, there will only be one time point. When both spatial_radius and
        temporal_radius are set to None, the result will be a single number (not packed
        in an SourceEstimate object). When multiple models have been supplied, an array
        will be returned containing the RSA results for each model.

    See Also
    --------
    compute_rdm
    """  # noqa E501
    # Check for compatibility of the source estimates and the model features
    one_model = type(rdm_model) is np.ndarray
    if one_model:
        rdm_model = [rdm_model]

    if labels_stcs is None and y is not None:
        labels_stcs = y

    # Check for compatibility of the stcs and the model features
    for rdm in rdm_model:
        n_items = _n_items_from_rdm(rdm)
        if len(stcs) != n_items and labels_stcs is None:
            raise ValueError(
                "The number of source estimates (%d) should be equal to the "
                "number of items in `rdm_model` (%d). Alternatively, use "
                "the `y` parameter to assign source estimates to items."
                % (len(stcs), n_items)
            )
        if labels_stcs is not None and len(set(labels_stcs)) != n_items:
            raise ValueError(
                "The number of items in `rdm_model` (%d) does not match "
                "the number of items encoded in the `labels_stcs` list (%d)."
                % (n_items, len(set(labels_stcs)))
            )

    _check_stcs_compatibility(stcs)
    if spatial_radius is not None:
        if src is None:
            raise ValueError(
                "When using `spatial_radius` to construct spatial searchlight patches, "
                "you also need to set `src` to the corresponding source space to "
                "allow distance calculations."
            )
        src = _check_src_compatibility(src, stcs[0])
        dist = _get_distance_matrix(src, dist_lim=spatial_radius, n_jobs=n_jobs)
    else:
        dist = None
    if temporal_radius is not None:
        # Convert the temporal radius to samples
        temporal_radius = int(temporal_radius // stcs[0].tstep)
        if temporal_radius < 1:
            raise ValueError("Temporal radius is less than one sample.")

    samples_from, samples_to = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    if sel_vertices_by_index is not None:
        sel_series = sel_vertices_by_index
    elif sel_vertices is None:
        sel_series = np.arange(len(stcs[0].data))
    else:
        sel_series = vertex_selection_to_indices(stcs[0].vertices, sel_vertices)
    if len(sel_series) != len(set(sel_series)):
        raise ValueError("Selected vertices are not unique. Please remove duplicates.")

    logger.info(
        f"Performing RSA between SourceEstimates and {len(rdm_model)} model RDM(s)"
    )
    if spatial_radius is not None:
        logger.info(f"    Spatial radius: {spatial_radius} meters")
    if sel_vertices is not None:
        logger.info(f"    Using {len(sel_series)} vertices")
    else:
        logger.info(f"    Using {sum(len(v) for v in stcs[0].vertices)} vertices")
    if temporal_radius is not None:
        logger.info(f"    Temporal radius: {temporal_radius} samples")
    if tmin is not None or tmax is not None:
        logger.info(f"    Time interval: {tmin}-{tmax} seconds")

    # Perform the RSA
    X = np.array([stc.data for stc in stcs])
    patches = searchlight(
        X.shape,
        dist=dist,
        spatial_radius=spatial_radius,
        temporal_radius=temporal_radius,
        sel_series=sel_series,
        samples_from=samples_from,
        samples_to=samples_to,
    )
    logger.info(f"    Number of searchlight patches: {len(patches)}")

    data = rsa_array(
        X,
        rdm_model,
        patches,
        data_rdm_metric=stc_rdm_metric,
        data_rdm_params=stc_rdm_params,
        rsa_metric=rsa_metric,
        ignore_nan=ignore_nan,
        labels_X=labels_stcs,
        labels_rdm_model=labels_rdm_model,
        n_folds=n_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if spatial_radius is None and temporal_radius is None:
        return data

    # Pack the result in a SourceEstimate object
    if spatial_radius is not None:
        vertices = vertex_indices_to_numbers(stcs[0].vertices, sel_series)
    else:
        if isinstance(stcs[0], mne.VolSourceEstimate):
            vertices = [np.array([1])]
        else:
            vertices = [np.array([1]), np.array([])]
        if one_model:
            data = data[np.newaxis, ...]
        else:
            data = data[:, np.newaxis, ...]
    tmin = _construct_tmin(stcs[0].times, samples_from, samples_to, temporal_radius)
    tstep = stcs[0].tstep

    if one_model:
        if isinstance(stcs[0], mne.VolSourceEstimate):
            return mne.VolSourceEstimate(
                data, vertices, tmin, tstep, subject=stcs[0].subject
            )
        else:
            return mne.SourceEstimate(
                data, vertices, tmin, tstep, subject=stcs[0].subject
            )
    else:
        if isinstance(stcs[0], mne.VolSourceEstimate):
            return [
                mne.VolSourceEstimate(
                    data[i], vertices, tmin, tstep, subject=stcs[0].subject
                )
                for i in range(data.shape[0])
            ]
        else:
            return [
                mne.SourceEstimate(
                    data[i], vertices, tmin, tstep, subject=stcs[0].subject
                )
                for i in range(data.shape[0])
            ]


@verbose
def rdm_stcs(
    stcs,
    src=None,
    spatial_radius=None,
    temporal_radius=None,
    dist_metric="sqeuclidean",
    dist_params=dict(),
    y=None,
    labels=None,
    n_folds=1,
    sel_vertices=None,
    sel_vertices_by_index=None,
    tmin=None,
    tmax=None,
    n_jobs=1,
    verbose=False,
):
    """Generate RDMs in a searchlight pattern on MNE-Python source estimates.

    RDMs are computed using a patch surrounding each source point. Source
    estimate objects can be either defined along a cortical surface
    (``SourceEstimate`` objects) or volumetric (``VolSourceEstimate`` objects).
    For surface source estimates, distances between vertices are measured in 2D
    space, namely as the length of the path along the surface from one vertex
    to another. For volume source estimates, distances are measured in 3D space
    as a straight line from one voxel to another.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate | list of mne.VolSourceEstimate
        For each item, a source estimate for the brain activity.
    src : instance of mne.SourceSpaces | None
        The source space used by the source estimates specified in the ``stcs``
        parameter. Only needed when ``spatial_radius`` is used to create spatial
        searchlight patches. Defaults to None.
    spatial_radius : float | None
        The spatial radius of the searchlight patch in meters. All source points within
        this radius will belong to the searchlight patch. When this is set, ``src`` also
        must be set to the correct source space. Set to None to only perform the
        searchlight over time, flattening across sensors. Defaults to None.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to None.
    dist_metric : str
        The metric to use to compute the RDM for the source estimates. This can
        be any metric supported by the scipy.distance.pdist function. See also
        the ``stc_rdm_params`` parameter to specify and additional parameter
        for the distance function. Defaults to 'sqeuclidean'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the RDMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        Deprecated, use ``labels`` instead.
        For each source estimate, a number indicating the item to which it
        belongs. Defaults to ``None``, in which case ``labels`` is used.
    labels : list | None
        For each source estimate, a label that identifies the item to which it
        corresponds. Multiple source estimates may correspond to the same item, in which
        case they should have the same label and will either be averaged when computing
        the data RDM (``n_folds=1``) or used for cross-validation (``n_folds>1``).
        Labels may be of any python type that can be compared with ``==`` (int, float,
        string, tuple, etc). By default (``None``), the integers ``0:len(evokeds)`` are
        used as labels.

        .. versionadded:: 0.10
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    sel_vertices :  mne.Label | list of mne.Label | list of ndarray | None
        When set, searchlight patches will only be generated for the subset of
        ROI labels, or vertices/voxels with the given vertex numbers. When
        giving vertex numbers, supply a list of numpy arrays with for each
        hemisphere, the selected vertex numbers. For volume source spaces,
        supple a list with only a single element, with that element being the
        ndarray with vertex numbers.
        See also ``sel_vertices_by_index`` for an alternative manner of
        selecting vertices.
    sel_vertices_by_index : ndarray of int, shape (n_vertices,)
        When set, searchlight patches will only be generated for the subset of
        vertices with the given indices in the ``stc.data`` array.
        See also ``sel_vertices`` for an alternative manner of selecting vertices.
    tmin : float | None
        When set, searchlight patches will only be generated from subsequent
        time points starting from this time point. This value is given in
        seconds. Defaults to ``None``, in which case patches are generated
        starting from the first time point.
    tmax : float | None
        When set, searchlight patches will only be generated up to and
        including this time point. This value is given in seconds. Defaults to
        ``None``, in which case patches are generated up to and including the
        last time point.
    n_jobs : int
        The number of processes (=number of CPU cores) to use for the
        source-to-source distance computation. Specify -1 to use all available
        cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Yields
    ------
    rdm : ndarray, shape (n_items, n_items)
        A RDM for each searchlight patch.

    """
    _check_stcs_compatibility(stcs)
    if spatial_radius is not None:
        if src is None:
            raise ValueError(
                "When using `spatial_radius` to construct spatial searchlight patches, "
                "you also need to set `src` to the corresponding source space to "
                "allow distance calculations."
            )
        src = _check_src_compatibility(src, stcs[0])
        dist = _get_distance_matrix(src, dist_lim=spatial_radius, n_jobs=n_jobs)
    else:
        dist = None

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = int(temporal_radius // stcs[0].tstep)

        if temporal_radius < 1:
            raise ValueError("Temporal radius is less than one sample.")

    samples_from, samples_to = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    if sel_vertices_by_index is not None:
        sel_series = sel_vertices_by_index
    elif sel_vertices is None:
        sel_series = np.arange(len(stcs[0].data))
    else:
        sel_series = vertex_selection_to_indices(stcs[0].vertices, sel_vertices)
    if len(sel_series) != len(set(sel_series)):
        raise ValueError("Selected vertices are not unique. Please remove duplicates.")

    if labels is None and y is not None:
        labels = y

    X = np.array([stc.data for stc in stcs])
    patches = searchlight(
        X.shape,
        dist=dist,
        spatial_radius=spatial_radius,
        temporal_radius=temporal_radius,
        sel_series=sel_series,
        samples_from=samples_from,
        samples_to=samples_to,
    )
    yield from rdm_array(
        X,
        patches,
        dist_metric=dist_metric,
        dist_params=dist_params,
        labels=labels,
        n_folds=n_folds,
        n_jobs=n_jobs,
    )


@verbose
def rsa_stcs_rois(
    stcs,
    rdm_model,
    src,
    rois,
    temporal_radius=None,
    stc_rdm_metric="correlation",
    stc_rdm_params=dict(),
    rsa_metric="spearman",
    ignore_nan=False,
    y=None,
    labels_stcs=None,
    labels_rdm_model=None,
    n_folds=1,
    tmin=None,
    tmax=None,
    n_jobs=1,
    verbose=False,
):
    """Perform RSA for a list of ROIs using MNE-Python source estimates.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computed for a patch surrounding the source point. Source estimate
    objects can be either defined along a cortical surface (``SourceEstimate``
    objects) or volumetric (``VolSourceEstimate`` objects).  For surface source
    estimates, distances between vertices are measured in 2D space, namely as
    the length of the path along the surface from one vertex to another. For
    volume source estimates, distances are measured in 3D space as a straight
    line from one voxel to another.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate | list of mne.VolSourceEstimate
        For each item, a source estimate for the brain activity.
    rdm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model RDM, see :func:`compute_rdm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model RDMs.

        Use :func:`compute_rdm` to compute RDMs.
    src : instance of mne.SourceSpaces
        The source space used by the source estimates specified in the `stcs`
        parameter.
    rois : list of mne.Label
        The spatial regions of interest (ROIs) to compute the RSA for. This
        needs to be specified as a list of ``mne.Label`` objects, such as
        returned by ``mne.read_annotations``.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to None.
    stc_rdm_metric : str
        The metric to use to compute the RDM for the source estimates. This can
        be any metric supported by the scipy.distance.pdist function. See also
        the ``stc_rdm_params`` parameter to specify and additional parameter
        for the distance function. Defaults to 'correlation'.
    stc_rdm_params : dict
        Extra arguments for the distance metric used to compute the RDMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    rsa_metric : str
        The RSA metric to use to compare the RDMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    y : ndarray of int, shape (n_items,) | None
        Deprecated, use ``labels_stcs`` and ``labels_rdm_model`` instead.
        For each source estimate, a number indicating the item to which it
        belongs. Defaults to ``None``, in which case ``labels_stcs`` is used.
    labels_stcs : list | None
        For each source estimate, a label that identifies the item to which it
        corresponds. This is used in combination with ``labels_rdm_model`` to align the
        data and model RDMs before comparing them. Multiple source estimates may
        correspond to the same item, in which case they should have the same label and
        will either be averaged when computing the data RDM (``n_folds=1``) or used for
        cross-validation (``n_folds>1``). Labels may be of any python type that can be
        compared with ``==`` (int, float, string, tuple, etc). By default (``None``),
        the integers ``0:len(evokeds)`` are used as labels.

        .. versionadded:: 0.10
    labels_rdm_model: list | None
        For each row in ``rdm_model``, a label that identifies the item to which it
        corresponds. This is used in combination with ``labels_stcs`` to align the
        data and model RDMs before comparing them. Each row should have a unique label.
        Labels may be of any python type that can be compared with ``==`` (int, float,
        string, tuple, etc). By default (``None``), the integers ``0:n_rows`` are used
        as labels.

        .. versionadded:: 0.10
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    tmin : float | None
        When set, searchlight patches will only be generated from subsequent
        time points starting from this time point. This value is given in
        seconds. Defaults to ``None``, in which case patches are generated
        starting from the first time point.
    tmax : float | None
        When set, searchlight patches will only be generated up to and
        including this time point. This value is given in seconds. Defaults to
        ``None``, in which case patches are generated up to and including the
        last time point.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    data : ndarray, shape (n_rois, n_times) | list of ndarray
        The correlation values for each ROI. When temporal_radius is set to
        None, there will be time dimension. When multiple models have been
        supplied, a list will be returned containing the RSA results for each
        model.
    stc : SourceEstimate | list of SourceEstimate
        The correlation values for each ROI, backfilled into a full
        SourceEstimate object. Each vertex belonging to the same ROI will have
        the same values. When temporal_radius is set to None, there will only
        be one time point. When multiple models have been supplied, a list will
        be returned containing the RSA results for each model.

    See Also
    --------
    compute_rdm

    """
    # Check for compatibility of the source estimates and the model features
    one_model = type(rdm_model) is np.ndarray
    if one_model:
        rdm_model = [rdm_model]

    if labels_stcs is None and y is not None:
        labels_stcs = y

    # Check for compatibility of the stcs and the model features
    for rdm in rdm_model:
        n_items = _n_items_from_rdm(rdm)
        if len(stcs) != n_items and labels_stcs is None:
            raise ValueError(
                "The number of source estimates (%d) should be equal to the "
                "number of items in `rdm_model` (%d). Alternatively, use "
                "the `labels_stcs` parameter to assign source estimates to items."
                % (len(stcs), n_items)
            )
        if labels_stcs is not None and len(set(labels_stcs)) != n_items:
            raise ValueError(
                "The number of items in `rdm_model` (%d) does not match "
                "the number of items encoded in the `labels_stcs` matrix (%d)."
                % (n_items, len(set(labels_stcs)))
            )

    _check_stcs_compatibility(stcs)
    src = _check_src_compatibility(src, stcs[0])

    if temporal_radius is not None:
        # Convert the temporal radius to samples
        temporal_radius = int(temporal_radius // stcs[0].tstep)

        if temporal_radius < 1:
            raise ValueError("Temporal radius is less than one sample.")

    samples_from, samples_to = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    # Convert the labels to data indices
    roi_inds = [vertex_selection_to_indices(stcs[0].vertices, roi) for roi in rois]

    # Perform the RSA
    X = np.array([stc.data for stc in stcs])
    patches = searchlight(
        X.shape,
        spatial_radius=roi_inds,
        temporal_radius=temporal_radius,
        samples_from=samples_from,
        samples_to=samples_to,
    )
    data = rsa_array(
        X,
        rdm_model,
        patches,
        data_rdm_metric=stc_rdm_metric,
        data_rdm_params=stc_rdm_params,
        rsa_metric=rsa_metric,
        ignore_nan=ignore_nan,
        labels_X=labels_stcs,
        labels_rdm_model=labels_rdm_model,
        n_folds=n_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Pack the result in SourceEstimate objects
    subject = stcs[0].subject
    tmin = _construct_tmin(stcs[0].times, samples_from, samples_to, temporal_radius)
    tstep = stcs[0].tstep
    if one_model:
        stc = backfill_stc_from_rois(
            data, rois, src, tmin=tmin, tstep=tstep, subject=subject
        )
    else:
        stc = [
            backfill_stc_from_rois(
                data[i], rois, src, tmin=tmin, tstep=tstep, subject=subject
            )
            for i in range(data.shape[0])
        ]

    return data, stc


@verbose
def rsa_nifti(
    image,
    rdm_model,
    spatial_radius=None,
    image_rdm_metric="correlation",
    image_rdm_params=dict(),
    rsa_metric="spearman",
    ignore_nan=False,
    y=None,
    labels_image=None,
    labels_rdm_model=None,
    n_folds=1,
    roi_mask=None,
    brain_mask=None,
    n_jobs=1,
    verbose=False,
):
    """Perform RSA in a searchlight pattern on Nibabel Nifti-like images.

    The output is a 3D Nifti image where the data at each voxel is is
    the RSA, computed for a patch surrounding the voxel.

    .. versionadded:: 0.4

    Parameters
    ----------
    image : 4D Nifti-like image
        The Nitfi image data. The 4th dimension must contain the images
        for each item.
    rdm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model RDM, see :func:`compute_rdm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model RDMs.

        Use :func:`compute_rdm` to compute RDMs.
    spatial_radius : float | None
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch.
        Defaults to ``None`` which will use a single searchlight patch.
    image_rdm_metric : str
        The metric to use to compute the RDM for the data. This can be
        any metric supported by the scipy.distance.pdist function. See also the
        ``image_rdm_params`` parameter to specify and additional parameter for
        the distance function. Defaults to 'correlation'.
    image_rdm_params : dict
        Extra arguments for the distance metric used to compute the RDMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    rsa_metric : str
        The RSA metric to use to compare the RDMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    y : ndarray of int, shape (n_items,) | None
        Deprecated, use ``labels_image`` and ``labels_rdm_model`` instead.
        For each image in the Nifti image object, a number indicating the item to which
        it belongs. Defaults to ``None``, in which case ``labels_image`` is used.
    labels_image : list | None
        For each image in the Nifti image object, a label that identifies the item to
        which it corresponds. This is used in combination with ``labels_rdm_model`` to
        align the data and model RDMs before comparing them. Multiple images objects may
        correspond to the same item, in which case they should have the same label and
        will either be averaged when computing the data RDM (``n_folds=1``) or used for
        cross-validation (``n_folds>1``). Labels may be of any python type that can be
        compared with ``==`` (int, float, string, tuple, etc). By default (``None``),
        the integers ``0:image.shape[3]`` are used as labels.

        .. versionadded:: 0.10
    labels_rdm_model: list | None
        For each row in ``rdm_model``, a label that identifies the item to which it
        corresponds. This is used in combination with ``labels_image`` to align the data
        and model RDMs before comparing them. Each row should have a unique label.
        Labels may be of any python type that can be compared with ``==`` (int, float,
        string, tuple, etc). By default (``None``), the integers ``0:n_rows`` are used
        as labels.

        .. versionadded:: 0.10
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    roi_mask : 3D Nifti-like image | None
        When set, searchlight patches will only be generated for the subset of
        voxels with non-zero values in the given mask. This is useful for
        restricting the analysis to a region of interest (ROI). Note that while
        the center of the patches are all within the ROI, the patch itself may
        extend beyond the ROI boundaries.
        Defaults to ``None``, in which case patches for all voxels are
        generated.
    brain_mask : 3D Nifti-like image | None
        When set, searchlight patches are restricted to only contain voxels
        with non-zero values in the given mask. This is useful for make sure
        only information from inside the brain is used. In contrast to the
        `roi_mask`, searchlight patches will not use data outside of this mask.
        Defaults to ``None``, in which case all voxels are included in the
        analysis.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa_results : 3D Nifti1Image | list of 3D Nifti1Image | float | ndarray
        The correlation values for each searchlight patch. When spatial_radius is set to
        None, the result will be a single number (not packed in an SourceEstimate
        object). When multiple models have been supplied, a list will be returned
        containing the RSA results for each model.

    See Also
    --------
    compute_rdm

    """
    # Check for compatibility of the source estimates and the model features
    one_model = type(rdm_model) is np.ndarray
    if one_model:
        rdm_model = [rdm_model]

    if (
        not isinstance(image, tuple(nib.imageclasses.all_image_classes))
        or image.ndim != 4
    ):
        raise ValueError("The image data must be 4-dimensional Nifti-like images")

    if labels_image is None and y is not None:
        labels_image = y

    # Check for compatibility of the BOLD images and the model features
    for rdm in rdm_model:
        n_items = _n_items_from_rdm(rdm)
        if image.shape[3] != n_items and labels_image is None:
            raise ValueError(
                "The number of images (%d) should be equal to the "
                "number of items in `rdm_model` (%d). Alternatively, use "
                "the `y` parameter to assign evokeds to items."
                % (image.shape[3], n_items)
            )
        if labels_image is not None and len(set(labels_image)) != n_items:
            raise ValueError(
                "The number of items in `rdm_model` (%d) does not match "
                "the number of items encoded in the `labels_image` list (%d)."
                % (n_items, len(set(labels_image)))
            )

    # Get data as (n_items x n_voxels)
    X = image.get_fdata().reshape(-1, image.shape[3]).T

    # Apply masks
    if spatial_radius is not None:
        result_mask = np.ones(image.shape[:3], dtype=bool)
    if brain_mask is not None:
        if brain_mask.ndim != 3 or brain_mask.shape != image.shape[:3]:
            raise ValueError(
                "Brain mask must be a 3-dimensional Nifi-like "
                "image with the same dimensions as the data "
                "image"
            )
        brain_mask = brain_mask.get_fdata() != 0
        if spatial_radius is not None:
            result_mask &= brain_mask
        brain_mask = brain_mask.ravel()
        X = X[:, brain_mask]
    if roi_mask is not None:
        if roi_mask.ndim != 3 or roi_mask.shape != image.shape[:3]:
            raise ValueError(
                "ROI mask must be a 3-dimensional Nifi-like "
                "image with the same dimensions as the data "
                "image"
            )
        roi_mask = roi_mask.get_fdata() != 0
        if spatial_radius is not None:
            result_mask &= roi_mask
        roi_mask = roi_mask.ravel()
        if brain_mask is not None:
            roi_mask = roi_mask[brain_mask]
        roi_mask = np.flatnonzero(roi_mask)

    # Compute distances between voxels
    if spatial_radius is not None:
        # Find voxel positions
        voxels = np.array(list(np.ndindex(image.shape[:-1])))
        voxel_loc = voxels @ image.affine[:3, :3]
        voxel_loc /= 1000  # convert position from mm to meters
        if brain_mask is not None:
            voxel_loc = voxel_loc[brain_mask]

        logger.info("Computing distances...")
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(radius=spatial_radius, n_jobs=n_jobs).fit(voxel_loc)
        dist = nn.radius_neighbors_graph(mode="distance")
    else:
        dist = None

    # Perform the RSA
    patches = searchlight(
        X.shape,
        dist=dist,
        spatial_radius=spatial_radius,
        temporal_radius=None,
        sel_series=roi_mask,
    )
    rsa_result = rsa_array(
        X,
        rdm_model,
        patches,
        data_rdm_metric=image_rdm_metric,
        data_rdm_params=image_rdm_params,
        rsa_metric=rsa_metric,
        ignore_nan=ignore_nan,
        labels_X=labels_image,
        labels_rdm_model=labels_rdm_model,
        n_folds=n_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if spatial_radius is None:
        return rsa_result

    if one_model:
        data = np.zeros(image.shape[:3])
        data[result_mask] = rsa_result
        return nib.Nifti1Image(data, image.affine, image.header)
    else:
        results = []
        for i in range(rsa_result.shape[0]):
            data = np.zeros(image.shape[:3])
            data[result_mask] = rsa_result[i]
            results.append(nib.Nifti1Image(data, image.affine, image.header))
        return results


@verbose
def rdm_nifti(
    image,
    spatial_radius=None,
    dist_metric="correlation",
    dist_params=dict(),
    y=None,
    labels=None,
    n_folds=1,
    roi_mask=None,
    brain_mask=None,
    n_jobs=1,
    verbose=False,
):
    """Generate RDMs in a searchlight pattern on Nibabel Nifty-like images.

    RDMs are computed using a patch surrounding each voxel.

    .. versionadded:: 0.4

    Parameters
    ----------
    image : 4D Nifti-like image
        The Nitfi image data. The 4th dimension must contain the images
        for each item.
    spatial_radius : float | None
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch.
        Defaults to ``None`` which will use a single searchlight patch.
    dist_metric : str
        The metric to use to compute the RDM for the data. This can be
        any metric supported by the scipy.distance.pdist function. See also the
        ``dist_params`` parameter to specify and additional parameter for
        the distance function. Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the RDMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        Deprecated, use ``labels`` instead.
        For each image in the Nifti image object, a number indicating the item to which
        it belongs. Defaults to ``None``, in which case ``labels`` is used.
    labels : list | None
        For each image in the Nifti image object, a label that identifies the item to
        which it corresponds. Multiple images objects may correspond to the same item,
        in which case they should have the same label and will either be averaged when
        computing the data RDM (``n_folds=1``) or used for cross-validation
        (``n_folds>1``). Labels may be of any python type that can be compared with
        ``==`` (int, float, string, tuple, etc). By default (``None``), the integers
        ``0:image.shape[3]`` are used as labels.

        .. versionadded:: 0.10
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    roi_mask : 3D Nifti-like image | None
        When set, searchlight patches will only be generated for the subset of
        voxels with non-zero values in the given mask. This is useful for
        restricting the analysis to a region of interest (ROI). Note that while
        the center of the patches are all within the ROI, the patch itself may
        extend beyond the ROI boundaries.
        Defaults to ``None``, in which case patches for all voxels are
        generated.
    brain_mask : 3D Nifti-like image | None
        When set, searchlight patches are restricted to only contain voxels
        with non-zero values in the given mask. This is useful for make sure
        only information from inside the brain is used. In contrast to the
        `roi_mask`, searchlight patches will not use data outside of this mask.
        Defaults to ``None``, in which case all voxels are included in the
        analysis.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Yields
    ------
    rdm : ndarray, shape (n_items, n_items)
        A RDM for each searchlight patch.

    """
    if (
        not isinstance(image, tuple(nib.imageclasses.all_image_classes))
        or image.ndim != 4
    ):
        raise ValueError("The image data must be 4-dimensional Nifti-like images")

    # Get data as (n_items x n_voxels)
    X = image.get_fdata().reshape(-1, image.shape[3]).T

    if labels is None and y is not None:
        labels = y

    # Find voxel positions
    voxels = np.array(list(np.ndindex(image.shape[:-1])))
    voxel_loc = voxels @ image.affine[:3, :3]
    voxel_loc /= 1000  # convert position from mm to meters

    # Apply masks
    if brain_mask is not None:
        if brain_mask.ndim != 3 or brain_mask.shape != image.shape[:3]:
            raise ValueError(
                "Brain mask must be a 3-dimensional Nifi-like "
                "image with the same dimensions as the data "
                "image"
            )
        brain_mask = brain_mask.get_fdata() != 0
        brain_mask = brain_mask.ravel()
        X = X[:, brain_mask]
        voxel_loc = voxel_loc[brain_mask, :]
    if roi_mask is not None:
        if roi_mask.ndim != 3 or roi_mask.shape != image.shape[:3]:
            raise ValueError(
                "ROI mask must be a 3-dimensional Nifi-like "
                "image with the same dimensions as the data "
                "image"
            )
        roi_mask = roi_mask.get_fdata() != 0
        roi_mask = roi_mask.ravel()
        if brain_mask is not None:
            roi_mask = roi_mask[brain_mask]
        roi_mask = np.flatnonzero(roi_mask)

    # Compute distances between voxels
    logger.info("Computing distances...")
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        radius=1e6 if spatial_radius is None else spatial_radius, n_jobs=n_jobs
    ).fit(voxel_loc)
    dist = nn.radius_neighbors_graph(mode="distance")

    # Compute RDMs
    patches = searchlight(
        X.shape,
        dist=dist,
        spatial_radius=spatial_radius,
        temporal_radius=None,
        sel_series=roi_mask,
    )
    yield from rdm_array(
        X,
        patches,
        dist_metric=dist_metric,
        dist_params=dist_params,
        labels=labels,
        n_folds=n_folds,
        n_jobs=n_jobs,
    )


def _check_stcs_compatibility(stcs):
    """Check for compatibility of the source estimates."""
    for stc in stcs:
        for vert1, vert2 in zip(stc.vertices, stcs[0].vertices):
            if len(vert1) != len(vert2) or np.any(vert1 != vert2):
                raise ValueError("Not all source estimates have the same vertices.")
    for stc in stcs:
        if len(stc.times) != len(stcs[0].times) or np.any(stc.times != stcs[0].times):
            raise ValueError("Not all source estimates have the same time points.")


def _check_src_compatibility(src, stc):
    """Check for compatibility of the source space with the given source estimate."""
    if isinstance(stc, mne.VolSourceEstimate) and src.kind != "volume":
        raise ValueError(
            "Volume source estimates provided, but not a volume source space "
            f"(src.kind={src.kind})."
        )
    if src.kind == "volume" and not isinstance(stc, mne.VolSourceEstimate):
        raise ValueError(
            "Volume source space provided, but not volume source estimates "
            f"(src.kind={src.kind})."
        )

    if src.kind == "volume":
        if len(np.setdiff1d(src[0]["vertno"], stc.vertices[0])) > 0:
            src = _restrict_src_to_vertices(src, stc.vertices)
    else:
        for src_hemi, stc_hemi_vertno in zip(src, stc.vertices):
            if len(np.setdiff1d(src_hemi["vertno"], stc_hemi_vertno)) > 0:
                src = _restrict_src_to_vertices(src, stc.vertices)
    return src


def _get_distance_matrix(src, dist_lim=None, n_jobs=1):
    """Get vertex-to-vertex distance matrix from source space.

    During inverse computation, the source space was downsampled (i.e. using
    ico4). Construct vertex-to-vertex distance matrices using only the
    vertices that are defined in the source solution.

    Parameters
    ----------
    src : mne.SourceSpaces
        The source space to get the distance matrix for.
    dist_lim : float | None
        Maximum distance required. We don't care about distances beyond this
        maximum. Defaults to None, which means an infinite distance limit.
    n_jobs : int
        Number of CPU cores to use if distance computation is necessary.
        Defaults to 1.

    Returns
    -------
    dist : ndarray (n_vertices, n_vertices)
        The vertex-to-vertex distance matrix.

    """
    dist = []
    if dist_lim is None:
        dist_lim = np.inf

    # Check if distances have been pre-computed in the given source space. Give
    # a warning if the pre-computed distances may have had a too limited
    # dist_lim setting.
    needs_distance_computation = False
    for hemi in src:
        if "dist" not in hemi or hemi["dist"] is None:
            needs_distance_computation = True
        else:
            if (
                hemi["dist_limit"] != np.inf
                and hemi["dist_limit"] >= 0
                and hemi["dist_limit"][0] < dist_lim
            ):
                warn(
                    f"Source space has pre-computed distances, but all "
                    f"distances are smaller than the searchlight radius "
                    f"({dist_lim}). You may want to consider recomputing "
                    f"the source space distances using the "
                    f"mne.add_source_space_distances function."
                )
                needs_distance_computation = True

    if needs_distance_computation:
        if src.kind == "volume":
            src = _add_volume_source_space_distances(src, dist_lim)
        else:
            src = mne.add_source_space_distances(src, dist_lim, n_jobs=n_jobs)

    for hemi in src:
        inuse = np.flatnonzero(hemi["inuse"])
        dist.append(hemi["dist"][np.ix_(inuse, inuse)].toarray())

    # Collect the distances in a single matrix
    dist = block_diag(*dist)
    dist[dist == 0] = np.inf  # Across hemisphere distance is infinity
    dist.flat[:: dist.shape[0] + 1] = 0  # Distance to yourself is zero

    return dist


def _add_volume_source_space_distances(src, dist_limit):
    """Compute the distance between voxels in a volume source space.

    Operates in-place!

    Code is mostly taken from `mne.add_source_space_distances`.

    Parameters
    ----------
    src : instance of mne.SourceSpaces
        The volume source space to compute the voxel-wise distances for.
    dist_limit : float
        The maximum distance (in meters) to consider. Voxels that are further
        apart than this distance will have a distance of infinity. Use this to
        reduce computation time.

    Returns
    -------
    src : instance of mne.SourceSpaces
        The volume source space, now with the 'dist' and 'dist_limit' fields
        set.

    """
    # Lazy import to not have to load the huge scipy module every time mne_rsa
    # gets loaded.
    from scipy.sparse import csr_matrix

    assert src.kind == "volume"
    n_sources = src[0]["np"]
    neighbors = np.array(src[0]["neighbor_vert"])
    rows, cols = np.nonzero(neighbors != -1)
    cols = neighbors[(rows, cols)]
    dist = np.linalg.norm(src[0]["rr"][rows, :] - src[0]["rr"][cols, :], axis=1)
    con_matrix = csr_matrix((dist, (rows, cols)), shape=(n_sources, n_sources))
    dist = mne.source_space._source_space._do_src_distances(
        con_matrix, src[0]["vertno"], np.arange(src[0]["nuse"]), dist_limit
    )[0]
    d = dist.ravel()  # already float32
    idx = d > 0
    d = d[idx]
    i, j = np.meshgrid(src[0]["vertno"], src[0]["vertno"])
    i = i.ravel()[idx]
    j = j.ravel()[idx]
    src[0]["dist"] = csr_matrix((d, (i, j)), shape=(n_sources, n_sources))
    src[0]["dist_limit"] = np.array([dist_limit], "float32")
    return src


def backfill_stc_from_rois(values, rois, src, tmin=0, tstep=1, subject=None):
    """Backfill the ROI values into a full mne.SourceEstimate object.

    Each vertex belonging to the same region of interest (ROI) will have the
    sample value.

    Parameters
    ----------
    values : ndarray, shape (n_rois, ...)
        For each ROI, either a single value or a timecourse of values.
    rois : list of mne.Label
        The spatial regions of interest (ROIs) to compute the RSA for. This
        needs to be specified as a list of ``mne.Label`` objects, such as
        returned by ``mne.read_annotations``.
    src : instance of mne.SourceSpaces
        The source space used by the source estimates specified in the `stcs`
        parameter.
    tmin : float
        Time corresponding to the first sample.
    tstep : float
        Difference in time between two samples.
    subject : str | None
        The name of the FreeSurfer subject.

    Returns
    -------
    stc : mne.SourceEstimate
        The backfilled source estimate object.

    """
    values = np.asarray(values)
    if values.ndim == 1:
        n_samples = 1
    else:
        n_samples = values.shape[1]
    data = np.zeros((src[0]["nuse"] + src[1]["nuse"], n_samples))
    verts_lh = src[0]["vertno"]
    verts_rh = src[1]["vertno"]
    for roi, rsa_timecourse in zip(rois, values):
        roi = roi.copy().restrict(src)
        if roi.hemi == "lh":
            roi_ind = np.searchsorted(verts_lh, roi.vertices)
        else:
            roi_ind = np.searchsorted(verts_rh, roi.vertices)
            roi_ind += src[0]["nuse"]
        for ind in roi_ind:
            data[ind] = rsa_timecourse
    return mne.SourceEstimate(
        data, vertices=[verts_lh, verts_rh], tmin=tmin, tstep=tstep, subject=subject
    )


def vertex_selection_to_indices(vertno, sel_vertices):
    """Unify across different ways of selecting vertices."""
    if isinstance(sel_vertices, mne.Label):
        sel_vertices = [sel_vertices]
    if not isinstance(sel_vertices, list):
        raise TypeError(
            "Invalid type for sel_vertices. It should be an mne.Label, a list of "
            f"mne.Label's or a list of numpy arrays, but {type(sel_vertices)} was "
            "provided."
        )
    if len(sel_vertices) == 0:
        raise ValueError("Empty list provided for sel_vertices.")

    # Deal with mne.Label objects
    if isinstance(sel_vertices[0], mne.Label):
        labels = sel_vertices
        sel_vertices = [[] for _ in range(len(vertno))]
        for label in labels:
            if label.hemi == "lh":
                sel_vertices[0].extend(label.get_vertices_used(vertno[0]).tolist())
            else:
                if len(vertno) == 1:
                    raise ValueError(
                        "one of the Label's provided for sel_vertices defines vertices "
                        "on the right hemisphere. However, the provided "
                        "SourceEstimate's only have vertices in the left hemisphere."
                    )
                sel_vertices[1].extend(label.get_vertices_used(vertno[1]).tolist())

    # At this point, sel_vertices should be an array of vertex numbers.
    # Convert them to indices for the .data array.
    if len(sel_vertices) != len(vertno):
        raise ValueError(
            f"sel_vertices should be a list with {len(vertno)} elements: one for "
            "each hemisphere (volume source spaces have one hemisphere.)"
        )
    sel_series = []
    for hemi, (hemi_vertno, hemi_sel_vertno) in enumerate(zip(vertno, sel_vertices)):
        if len(hemi_sel_vertno) == 0:
            continue
        sel_inds = np.searchsorted(hemi_vertno, hemi_sel_vertno)
        if not np.array_equal(hemi_vertno[sel_inds], hemi_sel_vertno):
            raise ValueError("Some selected vertices are not present in the data.")
        if hemi > 0:
            sel_inds += len(vertno[hemi - 1])
        sel_series.append(sel_inds)
    return np.unique(np.hstack(sel_series))


def vertex_indices_to_numbers(vertno, vert_ind):
    """Convert vertex indices to vertex numbers."""
    vert_ind = np.asarray(vert_ind)
    min_vert_ind = 0
    sel_vert_no = [[] for _ in vertno]
    for hemi, hemi_vertno in enumerate(vertno):
        if hemi > 0:
            min_vert_ind += len(vertno[hemi - 1])
        max_vert_ind = min_vert_ind + len(hemi_vertno)
        hemi_vert_ind = vert_ind[(vert_ind >= min_vert_ind) & (vert_ind < max_vert_ind)]
        hemi_vert_ind -= min_vert_ind
        sel_vert_no[hemi] = hemi_vertno[hemi_vert_ind]
    return sel_vert_no


@verbose
def _restrict_src_to_vertices(src, vertno, verbose=None):
    """Restrict a source space to the given vertices.

    Parameters
    ----------
    src: instance of SourceSpaces
        The source space to be restricted.
    vertno : tuple of lists (vertno_lh, vertno_rh)
        For each hemisphere, the vertex numbers to keep.
    verbose : bool | str | int | None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    src_out : instance of SourceSpaces
        The restricted source space.

    """
    assert isinstance(vertno, list) and (len(vertno) == 1 or len(vertno) == 2)
    src_out = deepcopy(src)

    for src_hemi, vertno_hemi in zip(src, vertno):
        if not (np.all(np.isin(vertno_hemi, src_hemi["vertno"]))):
            raise ValueError(
                "One or more vertices were not present in the source space."
            )
    logger.info(
        "Restricting source space to {n_keep} out of {n_total} vertices.".format(
            n_keep=sum(len(vertno_hemi) for vertno_hemi in vertno),
            n_total=np.hstack([src_hemi["nuse"] for src_hemi in src]),
        )
    )

    for hemi, verts in zip(src_out, vertno):
        # Ensure vertices are in sequential order
        verts = np.sort(verts)

        # Restrict the source space
        hemi["vertno"] = verts
        hemi["nuse"] = len(verts)
        hemi["inuse"] = hemi["inuse"].copy()
        hemi["inuse"].fill(0)
        if hemi["nuse"] > 0:  # Don't use empty array as index
            hemi["inuse"][verts] = 1
        hemi["use_tris"] = np.array([[]], int)
        hemi["nuse_tri"] = np.array([0])

    return src_out
