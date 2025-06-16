"""Classes and functions having to do with creating searchlights.

Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
"""

import numpy as np
from mne.utils import logger


class searchlight:
    """Generate indices for searchlight patches.

    Generates a sequence of tuples that can be used to index a data array.
    Depending on the spatial and temporal radius, each tuple extracts a
    searchlight patch along time, space or both.

    This function is flexible in regards to shape of the data array. The
    interpretation of the dimensions is as follows:

    4 or more dimensions
        ``(n_folds, n_items, n_series, n_samples, ...)``

    3 dimensions
        ``(n_items, n_series, n_samples)``

    2 dimensions
        ``(n_items, n_series)`` when ``spatial_radius`` or ``sel_series`` is set.

        ``(n_items, n_samples)`` when ``temporal_radius`` or
                ``samples_from``/``samples_to`` is set.

    1 dimension
        ``(n_items,)``

    The returned tuples will match the dimensions of the data array.

    Parameters
    ----------
    shape : tuple of int
        The shape of the data array to compute the searchlight patches for, as
        obtained with the ``.shape`` attribute.
    dist : ndarray or sparse matrix, shape (n_series, n_series) | None
        The distances between all source points or sensors in meters.
        This parameter needs to be specified if a ``spatial_radius`` is set.

        Since the distance matrix can be huge, sparse matrices are also
        supported. When the distance matrix is sparse, all zero distances are
        treated as infinity. This allows you to skip far away points during
        your distance computations.

        Defaults to ``None``.
    spatial_radius : float | list of list of int | None
        This controls how spatial patches will be created. There are several
        ways to do this:

        The first way is to specify a spatial radius in meters. In this case,
        the ``dist`` parameter must also be specified. This will create a
        searchlight where each patch contains all source points within this
        radius.

        The second way is to specify a list of predefined patches. In this
        case, each element of the list should itself be a list of integer
        indexes along the spatial dimension of the data array. Each element of
        this list will become a separate patch using the data at the specified
        indices.

        The third way is to set this to ``None``, which will disable the making
        of spatial patches and only perform the searchlight over time. This can
        be thought of as pooling everything into a single spatial patch.

        Defaults to``None``.
    temporal_radius : int | None
        The temporal radius of the searchlight patch in samples. Set to
        ``None`` to only perform the searchlight over sensors/source points.
        Defaults to ``None``.
    sel_series : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time series with the given indices. Defaults to ``None``, in which case
        patches for all series are generated.
    samples_from : int
        When set, searchlight patches will only be generated for the subset of
        time samples with indices equal or greater than the given value. Only
        used when the given data shape includes a temporal dimension.
        Defaults to 0.
    samples_to : int
        When set, searchlight patches will only be generated for the subset of
        time samples with indices up to, but not including, the given value.
        Only used when the given data shape includes a temporal dimension.
        Defaults to -1, which means there is no upper bound.

    Yields
    ------
    patch : tuple of (slice | ndarray)
        A single searchlight patch. Each element of the tuple corresponds to a
        dimension of the data array and can be used to index along this
        dimension to extract the searchlight patch.

    Attributes
    ----------
    shape

    """

    def __init__(
        self,
        shape,
        dist=None,
        spatial_radius=None,
        temporal_radius=None,
        sel_series=None,
        samples_from=0,
        samples_to=-1,
    ):
        # Interpret the dimensions of the data array (see docstring)
        n_dims = len(shape)
        if n_dims >= 4:
            self.series_dim = 2
            self.samples_dim = 3
        elif n_dims == 3:
            self.series_dim = 1
            self.samples_dim = 2
        elif n_dims == 2 and (spatial_radius is not None or sel_series is not None):
            self.series_dim = 1
            self.samples_dim = None
        elif n_dims == 2 and (
            temporal_radius is not None or samples_from != 0 or samples_to != -1
        ):
            self.series_dim = None
            self.samples_dim = 1
        else:
            self.series_dim = None
            self.samples_dim = None

        self.dist = dist
        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.sel_series = sel_series

        # Boundary checking for samples_from and samples_to. Only relevant if
        # there is a temporal dimension to the data.
        if samples_from != 0 or samples_to != -1:
            if self.samples_dim is None:
                raise ValueError(
                    "Cannot select samples: "
                    f"the provided data shape {shape} has no "
                    "temporal dimension."
                )
            n_samples = shape[self.samples_dim]
            if samples_from < 0 or samples_from > n_samples:
                raise ValueError(
                    f"`samples_from={samples_from}` is out "
                    f"of bounds given data shape ({shape})."
                )
            if samples_to > n_samples:
                raise ValueError(
                    f"`samples_to={samples_to}` is out "
                    f"of bounds given data shape ({shape})."
                )
            if samples_to != -1 and samples_to < samples_from:
                raise ValueError(
                    f"`samples_to={samples_to}` is smaller "
                    f"than `samples_from={samples_from}`."
                )
        self.samples_from = samples_from
        self.samples_to = samples_to

        # Will we be creating spatial searchlight patches?
        if self.spatial_radius is not None:
            if self.series_dim is None:
                raise ValueError(
                    "Cannot create spatial searchlight patches: "
                    f"the provided data shape ({shape}) has no "
                    "spatial dimension."
                )
            # If spatial radius is a number, we will be making searchlight
            # patches based on distance computations. Alternatively, a list of
            # predefined spatial patches may be provided, and we don't need
            # `dist`.
            if type(self.spatial_radius) in [float, int]:
                if self.dist is None:
                    raise ValueError(
                        "A spatial radius was requested, but no "
                        "distance information was specified "
                        "(=dist parameter)."
                    )
                # Compressed Sparse Row format is optimal for our computations
                from scipy.sparse import issparse

                if issparse(self.dist):
                    self.dist = self.dist.tocsr()
                if self.sel_series is None:
                    self.sel_series = np.arange(shape[self.series_dim])
            else:
                # Explicit spatial patches were provided
                if self.sel_series is None:
                    self.sel_series = np.arange(len(self.spatial_radius))

        # Will we be creating temporal searchlight patches?
        if temporal_radius is not None:
            if self.samples_dim is None:
                raise ValueError(
                    "Cannot create temporal searchlight patches: "
                    f"the provided data shape ({shape}) has no "
                    "temporal dimension."
                )
            n_samples = shape[self.samples_dim]

            # Compute the centers of the searchlight patches in time. Make sure
            # that adding/subtracting the temporal_radius does not produce
            # array out of bounds errors.
            samples_min = temporal_radius
            samples_max = n_samples - temporal_radius
            if samples_min > samples_max:
                raise ValueError(
                    f"Temporal radius ({temporal_radius}) too large for the "
                    f"given data shape ({shape})."
                )
            self.time_centers = list(
                range(
                    np.clip(samples_from, samples_min, samples_max),
                    np.clip(
                        n_samples if samples_to == -1 else samples_to,
                        samples_min,
                        samples_max,
                    ),
                )
            )

        # Create a template for the patches that will be generated that is
        # compatible with the data array dimensions. By default, we select
        # everything along every dimension, taking `sel_series`, `samples_from`
        # and `samples_to` into account. This template will be filled-in inside
        # the __iter__ function.
        self.patch_template = [slice(None)] * n_dims
        if self.sel_series is not None:
            if self.series_dim is None:
                raise ValueError(
                    "Cannot select series:"
                    f"the provided data shape {shape} has no "
                    "spatial dimension."
                )
            self.patch_template[self.series_dim] = self.sel_series
        if self.samples_from != 0 or self.samples_to != -1:
            # Bounds checking has already been done.
            self.patch_template[self.samples_dim] = slice(
                self.samples_from, self.samples_to
            )

        # Setup the main generator function that will be providing the
        # searchlight patches.
        self._generator = iter(self)

    def __iter__(self):
        """Get an iterator over the searchlight patches."""
        if self.spatial_radius is not None and self.temporal_radius is not None:
            iterator = self._iter_spatio_temporal()
        elif self.spatial_radius is not None:
            iterator = self._iter_spatial()
        elif self.temporal_radius is not None:
            iterator = self._iter_temporal()
        else:
            # Single searchlight patch only
            iterator = iter([tuple(self.patch_template)])
        return iterator

    def __next__(self):
        """Generate searchlight patches."""
        return next(self._generator)

    def _iter_spatio_temporal(self):
        """Generate spatio-temporal searchlight patches."""
        logger.info("Creating spatio-temporal searchlight patches")
        patch = list(self.patch_template)  # Copy the template
        for series in self.sel_series:
            # Compute all spatial locations in the searchligh path.
            if type(self.spatial_radius) in [float, int]:
                spat_ind = _get_in_radius(self.dist, series, self.spatial_radius)
            else:
                spat_ind = self.spatial_radius[series]
            patch[self.series_dim] = spat_ind
            for sample in self.time_centers:
                temp_ind = slice(
                    sample - self.temporal_radius, sample + self.temporal_radius + 1
                )
                patch[self.samples_dim] = temp_ind
                yield tuple(patch)

    def _iter_spatial(self):
        """Generate spatial searchlight patches only."""
        logger.info("Creating spatial searchlight patches")
        patch = list(self.patch_template)  # Copy the template
        for series in self.sel_series:
            if type(self.spatial_radius) in [float, int]:
                spat_ind = _get_in_radius(self.dist, series, self.spatial_radius)
            else:
                spat_ind = self.spatial_radius[series]
            patch[self.series_dim] = spat_ind
            yield tuple(patch)

    def _iter_temporal(self):
        """Generate temporal searchlight patches only."""
        logger.info("Creating temporal searchlight patches")
        patch = list(self.patch_template)  # Copy the template
        for sample in self.time_centers:
            patch[self.samples_dim] = slice(
                sample - self.temporal_radius, sample + self.temporal_radius + 1
            )
            yield tuple(patch)

    @property
    def shape(self):
        """Get the number of generated patches along multiple dimensions.

        This is useful for re-shaping the result obtained after consuming the
        this generator.

        Returns
        -------
        shape : tuple of int
            For a spatio-temporal searchlight:
                Two elements: the number of time-series and number of time
                samples for which patches are generated.
            For a spatial searchlight:
                One element: the number of time-series for which patches are
                generated.
            For a temporal searchlight:
                One element: the number of time-samples for which patches are
                generated.
            For no searchlight:
                Zero elements.

        """
        if self.spatial_radius is not None and self.temporal_radius is not None:
            return (len(self.sel_series), len(self.time_centers))
        elif self.spatial_radius is not None:
            return (len(self.sel_series),)
        elif self.temporal_radius is not None:
            return (len(self.time_centers),)
        else:
            return tuple()

    def __len__(self):
        """Get total number of searchlight patches that will be generated."""
        total = 1
        for n in self.shape:  # Number of patches generated in each dimension
            total *= n
        return total


def _get_in_radius(dist, seed, radius):
    """Obtain indices for all points within the given radius from a seed point.

    Takes care to work with sparse matrices too.

    Parameters
    ----------
    dist : ndarray or sparse matrix, shape (n_points, n_points)
        The distances between all points.
    seed : int
        The index of the point used as a seed.
    radius : float
        The maximum distance that points can be to be included.

    Returns
    -------
    ind : ndarray, shape (n_points_in_radius,)
        Indices of all points in the given radius from the seed point.

    """
    from scipy.sparse import issparse

    if issparse(dist):
        # Treat all zero distances as missing data
        ind = dist[seed].nonzero()[1]
        # Find indices for points within the radius
        ind = ind[dist[seed].data < radius]
        ind.sort()
        # Be sure to add the seed point, which has distance of 0 to itself
        ind = np.hstack((ind, [seed]))  # Di
    else:
        ind = np.flatnonzero(dist[seed] < radius)
    return sorted(ind)
