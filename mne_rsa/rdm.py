# encoding: utf-8
"""Methods to compute representational dissimilarity matrices (RDMs).

Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import distance

from .folds import create_folds, _match_order
from .searchlight import searchlight


def compute_rdm(data, metric="correlation", **kwargs):
    """Compute a dissimilarity matrix (RDM).

    Parameters
    ----------
    data : ndarray, shape (n_items, ...)
        For each item, all the features. The first dimension are the items and all other
        dimensions will be flattened and treated as features.
    metric : str | function
        The distance metric to use to compute the RDM. Can be any metric supported by
        :func:`scipy.spatial.distance.pdist`. When a function is specified, it needs to
        take in two vectors and output a single number. See also the ``dist_params``
        parameter to specify and additional parameter for the distance function.
        Defaults to 'correlation'.
    **kwargs : dict, optional
        Extra arguments for the distance metric. Refer to :mod:`scipy.spatial.distance`
        for a list of all other metrics and their arguments.

    Returns
    -------
    rdm : ndarray, shape (n_classes * n_classes-1,)
        The RDM, in condensed form. See :func:`scipy.spatial.distance.squareform`.

    See Also
    --------
    compute_rdm_cv

    """
    X = np.reshape(np.asarray(data), (len(data), -1))
    n_items, n_features = X.shape

    # Be careful with certain metrics
    if n_features == 1 and metric in ["correlation", "cosine"]:
        raise ValueError(
            "There is only a single feature, so "
            "'correlation' and 'cosine' can not be "
            "used as RDM metric. Consider using 'sqeuclidean' "
            "instead."
        )

    return distance.pdist(X, metric, **kwargs)


def compute_rdm_cv(folds, metric="correlation", **kwargs):
    """Compute a dissimilarity matrix (RDM) using cross-validation.

    The distance computation is performed from the average of all-but-one "training"
    folds to the remaining "test" fold. This is repeated with each fold acting as the
    "test" fold once. The resulting distances are averaged and the result used in the
    final RDM.

    Parameters
    ----------
    folds : ndarray, shape (n_folds, n_items, ...)
        For each item, all the features. The first dimension are the folds used for
        cross-validation, items are along the second dimension, and all other dimensions
        will be flattened and treated as features.
    metric : str | function
        The distance metric to use to compute the RDM. Can be any metric supported by
        :func:`scipy.spatial.distance.pdist`. When a function is specified, it needs to
        take in two vectors and output a single number. See also the ``dist_params``
        parameter to specify and additional parameter for the distance function.
        Defaults to 'correlation'.
    **kwargs : dict, optional
        Extra arguments for the distance metric. Refer to :mod:`scipy.spatial.distance`
        for a list of all other metrics and their arguments.

    Returns
    -------
    rdm : ndarray, shape (n_classes * n_classes-1,)
        The cross-validated RDM, in condensed form.
        See :func:`scipy.spatial.distance.squareform`.

    See Also
    --------
    compute_rdm

    """
    X = np.reshape(folds, (folds.shape[0], folds.shape[1], -1))
    n_folds, n_items, n_features = X.shape[:3]

    # Be careful with certain metrics
    if n_features == 1 and metric in ["correlation", "cosine"]:
        raise ValueError(
            "There is only a single feature, so "
            "'correlataion' and 'cosine' can not be "
            "used as RDM metric. Consider using 'sqeuclidean' "
            "instead."
        )

    rdm = np.zeros((n_items * (n_items - 1)) // 2)

    X_mean = X.mean(axis=0)

    # Do cross-validation
    for test_fold in range(n_folds):
        X_test = X[test_fold]
        X_train = X_mean - (X_mean - X_test) / (n_folds - 1)

        dist = distance.cdist(X_train, X_test, metric, **kwargs)
        rdm += dist[np.triu_indices_from(dist, 1)]

    return rdm / n_folds


def _ensure_condensed(rdm, var_name):
    """Convert a RDM to condensed form if needed."""
    if isinstance(rdm, list):
        return [_ensure_condensed(d, var_name) for d in rdm]

    if not isinstance(rdm, np.ndarray):
        raise TypeError(
            "A single RDM should be a NumPy array. "
            "Multiple RDMs should be a list of NumPy arrays."
        )

    if rdm.ndim == 2:
        if rdm.shape[0] != rdm.shape[1]:
            raise ValueError(
                f'Invalid dimensions for "{var_name}" '
                "({rdm.shape}). The RDM should either be a "
                "square matrix, or a one dimensional array when "
                "in condensed form."
            )
        rdm = distance.squareform(rdm)
    elif rdm.ndim != 1:
        raise ValueError(
            f'Invalid dimensions for "{var_name}" ({rdm.shape}). '
            "The RDM should either be a square matrix, or a one "
            "dimensional array when in condensed form."
        )
    return rdm


def _n_items_from_rdm(rdm):
    """Get the number of items, given a RDM."""
    if rdm.ndim == 2:
        return rdm.shape[0]
    elif rdm.ndim == 1:
        return distance.squareform(rdm).shape[0]
    else:
        raise ValueError(f"Wrong number of dimensions for RDM ({rdm.ndim})")


def pick_rdm(rdm, sel):
    """Select items from an RDM.

    Parameters
    ----------
    rdm : ndarray, shape (n, n) | (n * (n - 1) // 2,)
        The RDM to pick items from.
    sel : int | list of int | boolean mask | slice
        The items to pick. These items will be selected from both rows and columns of
        the RDM.

    Returns
    -------
    rdm : ndarray, shape (n, n) | (n * (n - 1) // 2,)
        A new RDM with only the selected items. If the original RDM was in condensed
        form, the returned RDM will be as well.

    """
    if np.isscalar(sel):
        sel = [sel]  # to avoid dropping a dimension
    if rdm.ndim == 2:
        return rdm[sel, :][:, sel]
    elif rdm.ndim == 1:
        return distance.squareform(distance.squareform(rdm)[sel, :][:, sel])
    else:
        raise ValueError(f"Wrong number of dimensions for RDM ({rdm.ndim})")


class rdm_array:
    """Generate RDMs from an array of data, possibly in a searchlight pattern.

    First use :class:`searchlight` to compute the searchlight patches. Then you can use
    this function to compute RDMs for each searchlight patch.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, n_times)
        An array containing the data.
    patches : generator of tuples | None
        Searchlight patches as generated by :class:`searchlight`. If ``None``, no
        searchlight is used. Defaults to ``None``.
    dist_metric : str | function
        The distance metric to use to compute the RDMs. Can be any metric supported by
        :func:`scipy.spatial.distance.pdist`. When a function is specified, it needs to
        take in two vectors and output a single number. See also the ``dist_params``
        parameter to specify and additional parameter for the distance function.
        Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the RDMs. Refer to
        :mod:`scipy.spatial.distance` for a list of all other metrics and their
        arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        Deprecated, use ``labels`` instead.
        For each item, a number indicating the class to which the item belongs.
        Defaults to ``None``, in which case ``labels`` is used.
    labels : list | None
        For each element in ``X`` (=the first dimension), a label that identifies the
        item to which it corresponds. Multiple elements in ``X`` may correspond to the
        same item, in which case they should have the same label and will either be
        averaged when computing the data RDM (``n_folds=1``) or used for
        cross-validation (``n_folds>1``). Labels may be of any python type that can be
        compared with ``==`` (int, float, string, tuple, etc). By default (``None``),
        the integers ``0:len(X)`` are used as labels.

        .. versionadded:: 0.10
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance metric.
        Folds are created based on the ``y`` parameter. Specify ``None`` to use the
        maximum number of folds possible, given the data. Alternatively, you can pass a
        Scikit-Learn cross validator object (e.g. ``sklearn.model_selection.KFold``) to
        assert fine-grained control over how folds are created.
        Defaults to 1 (no cross-validation).

    Yields
    ------
    rdm : ndarray, shape (n_patches, n_items * (n_items - 1))
        A RDM (in condensed form) for each searchlight patch.

    Attributes
    ----------
    shape : tuple of int
        Multidimensional shape of the generated RDMs.

        This is useful for re-shaping the result obtained after consuming the this
        generator.

        For a spatio-temporal searchlight:
            Three elements: the number of time-series, number of time samples and length
            of a consensed RDM.
        For a spatial searchlight:
            Two element: the number of time-series and length of a condensed RDM.
        For a temporal searchlight:
            Two elements: the number of time-samples and length of a condensed RDM.
        For no searchlight:
            One element: the length of a condensed RDM.

    See Also
    --------
    rdm
    rsa
    searchlight

    """

    def __init__(
        self,
        X,
        patches=None,
        dist_metric="correlation",
        dist_params=dict(),
        y=None,
        labels=None,
        n_folds=1,
        n_jobs=1,
    ):
        if patches is None:
            patches = searchlight(X.shape)

        if y is None:
            y = _match_order(len_X=len(X), labels_X=labels, var="labels")

        # Create folds for cross-validated RDM metrics
        self.X = create_folds(X, y, n_folds)
        # The data is now folds x items x n_series x ...

        self.patches = patches
        self.dist_metric = dist_metric
        self.dist_params = dist_params
        self.use_cv = len(self.X) > 1  # More than one fold present

        # Target shape for an array that would hold all of the generated RDMs.
        rdm_length = len(np.triu_indices(self.X.shape[1], k=1)[0])
        self.shape = patches.shape + (rdm_length,)

        self.n_jobs = n_jobs

        # Setup the generator that will be producing the RDMs
        self._generator = iter(self)

    def __iter__(self):
        """Build a new iterator that starts from the beginning."""
        return self._iter_rdms()

    def __next__(self):
        """Generate a RDM for each patch."""
        return next(self._generator)

    def _iter_rdms(self):
        par = Parallel(n_jobs=self.n_jobs, return_as="generator")
        if self.use_cv:
            yield from par(
                delayed(compute_rdm_cv)(
                    self.X[(slice(None),) + patch], self.dist_metric, **self.dist_params
                )
                for patch in self.patches
            )
        else:
            yield from par(
                delayed(compute_rdm)(
                    self.X[0][patch], self.dist_metric, **self.dist_params
                )
                for patch in self.patches
            )

    def __len__(self):
        """Get total number of RDMs that will be generated."""
        return len(self.patches)
