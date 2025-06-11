"""Functions concerning the creation of cross-validation folds.

Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
Yuan-Fang Zhao <distancejay@gmail.com>
"""

import numpy as np
from mne.utils import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def create_folds(X, y=None, n_folds=None):
    """Group individual items into folds suitable for cross-validation.

    The ``y`` list should contain an integer label for each item in ``X``. Repetitions
    of the same item have the same integer label. Repeated items are distributed evenly
    across the folds, and averaged within a fold.

    Parameters
    ----------
    X : ndarray, shape (n_items, ...)
        For each item, all the features. The first dimension are the items and all other
        dimensions will be flattened and treated as features.
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs. When
        ``None``, each item is assumed to belong to a different class. Defaults to
        ``None``.
    n_folds : int | sklearn.BaseCrossValidator | None
        Number of cross-validation folds to use when computing the distance metric.
        Folds are created based on the ``y`` parameter. Specify ``None`` to use the
        maximum number of folds possible, given the data. Alternatively, you can pass a
        Scikit-Learn cross validator object (e.g. ``sklearn.model_selection.KFold``) to
        assert fine-grained control over how folds are created. Defaults to ``None``.

    Returns
    -------
    folds : ndarray, shape (n_folds, n_items, ...)
        The folded data.

    """
    if y is None:
        # No folding
        return X[np.newaxis, ...]

    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError(
            f"The length of y ({len(y)}) does not match the "
            f"number of items ({len(X)})."
        )

    y_one_hot = _convert_to_one_hot(y)
    n_items = y_one_hot.shape[1]

    if n_folds is None:
        # Set n_folds to maximum value
        n_folds = len(X) // n_items
        logger.info(
            f"Automatic dermination of folds: {n_folds}" + " (no cross-validation)"
            if n_folds == 1
            else ""
        )

    if n_folds == 1:
        # Making one fold is easy
        folds = [_compute_item_means(X, y_one_hot)]
    elif hasattr(n_folds, "split"):
        # Scikit-learn object passed as `n_folds`
        folds = []
        for _, fold in n_folds.split(X, y):
            folds.append(_compute_item_means(X, y_one_hot, fold))
    else:
        # Use StratifiedKFold as folding strategy
        folds = []
        for _, fold in StratifiedKFold(n_folds).split(X, y):
            folds.append(_compute_item_means(X, y_one_hot, fold))
    return np.array(folds)


def _convert_to_one_hot(y):
    """Convert the labels in y to one-hot encoding."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, np.newaxis]

    if y.ndim == 2 and y.shape[1] == 1:
        # y needs to be converted
        enc = OneHotEncoder(categories="auto").fit(y)
        return enc.transform(y).toarray()
    elif y.ndim > 2:
        raise ValueError("Wrong number of dimensions for `y`.")
    else:
        # y is probably already in one-hot form. We're not going to test this
        # explicitly, as it would take too long.
        return y


def _compute_item_means(X, y_one_hot, fold=slice(None)):
    """Compute the mean data for each item inside a fold."""
    X = X[fold]
    y_one_hot = y_one_hot[fold]
    n_per_class = y_one_hot.sum(axis=0)

    # The following computations go much faster when X is flattened.
    orig_shape = X.shape
    X_flat = X.reshape(len(X), -1)

    # Compute the mean for each item using matrix multiplication
    means = (y_one_hot.T @ X_flat) / n_per_class[:, np.newaxis]

    # Undo the flattening of X
    return means.reshape((len(means),) + orig_shape[1:])
